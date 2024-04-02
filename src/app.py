from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_groq import ChatGroq


def initialize_database_connection(username: str, password: str, host: str, port: str, database_name: str) -> SQLDatabase:
    """
    Initialize a connection to the SQL database using the provided credentials.

    Args:
        username (str): Username for database authentication.
        password (str): Password for database authentication.
        host (str): Database host address.
        port (str): Port number for database connection.
        database_name (str): Name of the database.

    Returns:
        SQLDatabase: A connection to the SQL database.
    """
    # Construct the URI for connecting to the database
    db_uri = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database_name}"
    
    # Create a connection to the database using the URI
    return SQLDatabase.from_uri(db_uri)


def generate_sql_query_chain(database):
    """
    Generate a pipeline to interactively generate SQL queries based on conversation history.

    Args:
        database: An object representing the database.

    Returns:
        Pipeline: A pipeline for generating SQL queries.
    """
    # Template for generating SQL queries
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}

        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

        For example:
        Question: which 3 artists have the most tracks?
        SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
        Question: Name 10 artists
        SQL Query: SELECT Name FROM Artist LIMIT 10;

        Your turn:

        Question: {question}
        SQL Query:
        """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # llm = ChatOpenAI(model="gpt-4-0125-preview")
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    # llm = ChatGroq(model="llama2-70b-4096", temperature=0)
    
    def retrieve_schema(_):
        return database.get_table_info()
    
    # Pipeline for generating SQL queries
    return (
        RunnablePassthrough.assign(schema=retrieve_schema)
        | prompt
        | llm
        | StrOutputParser()
    )


def generate_response(user_query: str, database: SQLDatabase, chat_history: list):
    """
    Generate a response to a user query based on conversation history and SQL database interactions.

    Args:
        user_query (str): User's query.
        database (SQLDatabase): Object representing the SQL database.
        chat_history (list): List of previous conversation history.

    Returns:
        str: Natural language response to the user query.
    """
    
    sql_query_pipeline = generate_sql_query_chain(database)
    
    # Template for generating response
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # llm = ChatOpenAI(model="gpt-4-0125-preview")
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    # llm = ChatGroq(model="llama2-70b-4096", temperature=0)
    
    # Pipeline for generating responses
    response_chain = (
        RunnablePassthrough.assign(query=sql_query_pipeline).assign(
            schema=lambda _: database.get_table_info(),
            response=lambda vars: database.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Invoke the response chain with relevant variables
    return response_chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

# Environment variables
load_dotenv()

# App configuration
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")
st.title("Chat with MySQL")

# Sidebar for database connection settings
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")

    host = st.text_input("Host", value="localhost", key="Host")
    port = st.text_input("Port", value="3306", key="Port")
    user = st.text_input("User", value="root", key="User")
    password = st.text_input("Password", type="password", value="admin", key="Password")
    database = st.text_input("Database", value="Chinook", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = initialize_database_connection(
                user=user,
                password=password,
                host=host,
                port=port,
                database=database
            )
            st.session_state.db = db
            st.success("Connected to database!")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Input field for user query
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Display user message
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Generate and display AI response
    with st.chat_message("AI"):
        response = generate_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)

    # Add AI response to chat history
    st.session_state.chat_history.append(AIMessage(content=response))
