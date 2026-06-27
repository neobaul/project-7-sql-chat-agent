# SQL Chat Agent — Enterprise AI Analytics

A Streamlit app that lets you chat with any SQL database in plain English. Ask questions about your data and get answers, live data tables, and auto-generated visualizations — no SQL knowledge required.

---

## What It Does

Type a question like *"Show me transaction amounts by card type"* and the agent:
1. Translates your question into a SQL query
2. Runs it against your database
3. Returns the answer in plain English
4. Displays the raw results as a DataFrame
5. Generates and executes Python chart code (matplotlib/seaborn) to visualize the data automatically

The agent's full thought process (Thought → Action → Observation loop) is visible in an expandable panel on every response.

---

## Project Structure

```
├── sql_chat_agent.py    # Main Streamlit application
├── creditcard.db        # Sample SQLite database (credit card transactions)
└── .gitignore
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/neobaul/project-7-sql-chat-agent.git
cd project-7-sql-chat-agent
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install streamlit pandas sqlalchemy matplotlib seaborn
pip install langchain-community langchain-core langchain-groq langchain-openai
pip install langchain-google-genai langchain-ollama mysql-connector-python
```

### 4. Set up API keys
Create a `.env` file or enter your keys directly in the sidebar:
```
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here    # for Gemini
```

### 5. Run the app
```bash
streamlit run sql_chat_agent.py
```

---

## How It Works

```
User question (natural language)
    ↓
SQL Agent (ReAct loop)
  → Inspects database schema
  → Writes SQL query
  → Executes query
  → Interprets results
    ↓
Plain English answer + raw DataFrame
    ↓
Chart Agent
  → Analyzes the DataFrame structure
  → Generates matplotlib/seaborn Python code
  → Executes code and renders chart inline
```

Two separate agents work in sequence: the SQL agent handles data retrieval, and a second charting agent handles visualization — each with its own specialized prompt.

---

## Configuration (Sidebar)

**Step 1 — Database Connection**
- **SQLite** — point to a local `.db` file (defaults to `creditcard.db`)
- **MySQL** — enter host, port, username, password, and database name

**Step 2 — AI Agent Setup** (appears after connecting)
- Provider: Groq, OpenAI, Google, or Ollama
- Model name: e.g., `llama3-70b-8192`, `gpt-4o`, `gemini-1.5-pro`
- API key field (auto-disabled for Ollama)

---

## Example Queries

- *"How many transactions are there per card type?"*
- *"What is the average transaction amount?"*
- *"Show me the top 10 largest transactions."*
- *"Which merchant category has the most fraud?"*

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web App | Streamlit |
| SQL Agent | LangChain `create_sql_agent` (ReAct / ZERO_SHOT_REACT) |
| Chart Agent | LangChain LCEL prompt chain |
| LLM Providers | Groq, OpenAI, Google Gemini, Ollama |
| Database Support | SQLite, MySQL (via SQLAlchemy) |
| Visualization | Matplotlib + Seaborn (AI-generated code) |
| Language | Python 3.11 |
