# --- Import Libraries ---
import streamlit as st
import pandas as pd
from pathlib import Path
import sqlite3
from urllib.parse import quote_plus
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Database Import ---
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# --- LangChain Core Import ---
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LLM Provider Import ---
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# --- Set Page Config ---
def main():
    """
    Main function to run the Streamlit application.
    """

    st.set_page_config(
        page_title="Enterprise AI Analytics Agent",
        page_icon="🧠",
        layout="wide"
    )
    st.title("🧠 Enterprise AI Analytics Agent")
    st.caption("Your AI-powered partner for SQL queries and automated data visualization.")

    init_session_state()
    configure_sidebar()
    display_chat_interface()

def init_session_state():
    """Initializes all required keys in Streamlit's session state."""
    defaults = {
        "agent_executor": None,
        "chart_agent": None,
        "db": None,
        "db_connection_status": "Not Connected",
        "db_error": None,
        "messages": [{"role": "assistant", "content": "Welcome! Please configure your database and LLM in the sidebar to begin."}]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_llm(_provider, _api_key, _model_name):
    """Factory function to get an instance of the selected LLM."""
    try:
        if _provider == "Groq":
            return ChatGroq(groq_api_key=_api_key, model_name=_model_name, temperature=0)
        elif _provider == "OpenAI":
            return ChatOpenAI(api_key=_api_key, model_name=_model_name, temperature=0)
        elif _provider == "Google":
            return ChatGoogleGenerativeAI(google_api_key=_api_key, model_name=_model_name, temperature=0)
        elif _provider == "Ollama":
            return ChatOllama(model=_model_name, temperature=0)
    except Exception as e:
        st.sidebar.error(f"LLM Error: {e}")
    return None


def create_agents(llm, db):
    """Creates and configures the SQL and Charting agents."""
    # --- SQL Agent ---
    SQL_AGENT_PROMPT_TEMPLATE = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results.
Unless the user specifies a specific number of examples to obtain, query for at most {top_k} results.
You have access to the following tools:
{tools}

Use the following format for your response:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
    prompt = PromptTemplate.from_template(template=SQL_AGENT_PROMPT_TEMPLATE)
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_executor_kwargs={"return_intermediate_steps": True},
        prompt=prompt
    )
    st.session_state.agent_executor = agent_executor

    # --- Charting Agent ---
    chart_prompt = ChatPromptTemplate.from_template(
        """
        You are an expert data visualization assistant. Generate Python code to create a chart based on a user question.

        **Instructions:**
        1. Analyze the user's question: {question}
        2. The dataframe `df` has the following structure: {df_info}
        3. Choose the most appropriate chart type (bar, line, scatter, etc.).
        4. Generate **only the Python code** for the visualization using `matplotlib.pyplot` and `seaborn`.
        5. **IMPORTANT**: Use double quotes (") for all strings, especially for titles and labels, to avoid syntax errors.
        6. Use the exact column names from the dataframe structure above.
        7. A figure and axes are already defined as `fig` and `ax` — use them directly.
        8. **Do not** include `plt.show()`, `plt.subplots()`, or any import statements.
        9. **Do not** output any text or markdown formatting outside the code block.
        """
    )
    chart_agent = chart_prompt | llm | StrOutputParser()
    st.session_state.chart_agent = chart_agent


def connect_to_db(db_type, config):
    """Establishes and tests the database connection."""
    try:
        if db_type == "SQLite":
            db_file = Path(__file__).parent / config["db_name"]
            if not db_file.exists():
                raise FileNotFoundError(f"SQLite file not found at: {db_file}")
            creator = lambda: sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            engine = create_engine("sqlite:///", creator=creator)
        else: # MySQL
            password = quote_plus(config["password"])
            uri = f"mysql+mysqlconnector://{config['user']}:{password}@{config['host']}:{config['port']}/{config['database']}"
            engine = create_engine(uri, pool_recycle=3600)

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        st.session_state.db_connection_status = "Success"
        st.session_state.db_error = None
        return SQLDatabase(engine)

    except (SQLAlchemyError, FileNotFoundError) as e:
        error_msg = str(e).split('\n')[0]
        st.session_state.db_connection_status = "Error"
        st.session_state.db_error = f"Connection Failed: {error_msg}"
        return None


def configure_sidebar():
    """Sets up the sidebar for database and LLM configuration."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("1. Database Connection")
        db_type = st.radio("Database Type", ("SQLite", "MySQL"), key="db_type")

        db_config = {"type": db_type}
        if db_type == "SQLite":
            db_config["db_name"] = st.text_input("DB Filename", "creditcard.db")
        else: # MySQL
            db_config["host"] = st.text_input("Host", "localhost")
            db_config["port"] = st.text_input("Port", "3306")
            db_config["user"] = st.text_input("Username", "root")
            db_config["password"] = st.text_input("Password", type="password")
            db_config["database"] = st.text_input("Database Name")

        if st.button("Connect to Database"):
            with st.spinner("Connecting..."):
                st.session_state.db = connect_to_db(db_type, db_config)

        if st.session_state.db_connection_status == "Success":
            st.success("Database connection successful!")
        elif st.session_state.db_connection_status == "Error":
            st.error(st.session_state.db_error)

        st.divider()

        if st.session_state.db:
            st.subheader("2. AI Agent Setup")
            llm_provider = st.selectbox("LLM Provider", ["Groq", "OpenAI", "Google", "Ollama"])
            api_key = ""
            if llm_provider != "Ollama":
                api_key = st.text_input(f"{llm_provider} API Key", type="password")
            model_name = st.text_input("Model Name", "gpt-4o-mini", help="e.g., llama3-70b-8192, gpt-4o, gemini-1.5-pro")

            if st.button("Create Agents"):
                if (llm_provider != "Ollama" and not api_key) or not model_name:
                    st.warning("Please provide all LLM details.")
                else:
                    with st.spinner("Creating AI agents..."):
                        llm = get_llm(llm_provider, api_key, model_name)
                        if llm:
                            create_agents(llm, st.session_state.db)
                            st.success("AI agents are ready!")
                            st.session_state.messages = [{"role": "assistant", "content": "Agents created. How can I help you analyze the the database?"}]


def display_chat_interface():
    """Handles the main chat display and user interaction logic."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg and msg["df"] is not None:
                st.dataframe(msg["df"])
            if "chart_code" in msg:
                with st.expander("📊 Generated Chart Code"):
                    st.code(msg["chart_code"], language="python")
            if "fig" in msg and msg["fig"] is not None:
                st.pyplot(msg["fig"])

    if user_query := st.chat_input("Ask about your data... e.g., 'Show me transaction amounts by card type'"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        if not st.session_state.agent_executor:
            st.info("Please create the AI agents in the sidebar first.")
            st.stop()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent_executor.invoke({"input": user_query})
                    final_answer = response["output"]

                    with st.expander("🔍 Agent's Thought Process"):
                        st.write(response["intermediate_steps"])

                    st.markdown(final_answer)

                    df = None
                    sql_query = next((step[0].tool_input for step in reversed(response["intermediate_steps"])), None)

                    if sql_query:
                        with st.session_state.db._engine.connect() as connection:
                            df = pd.read_sql_query(text(sql_query), connection)

                    assistant_message = {"role": "assistant", "content": final_answer}

                    if df is not None and not df.empty:
                        st.dataframe(df)
                        with st.spinner("Generating visualization..."):
                            df_info = f"Columns: {list(df.columns)}\nSample:\n{df.head(3).to_string()}"
                            chart_code_str = st.session_state.chart_agent.invoke({"question": user_query, "df_info": df_info})
                            chart_code = chart_code_str.strip().replace("```python", "").replace("```", "")

                            with st.expander("📊 Generated Chart Code"):
                                st.code(chart_code, language="python")

                            assistant_message["chart_code"] = chart_code

                            fig, ax = plt.subplots()
                            exec_globals = {'plt': plt, 'sns': sns, 'df': df, 'fig': fig, 'ax': ax}
                            exec(chart_code, exec_globals)
                            st.pyplot(fig)

                            assistant_message.update({"df": df, "fig": fig})

                    st.session_state.messages.append(assistant_message)

                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()
