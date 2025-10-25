# Plan for Stock Advisory System Backend
 
## Overview
This plan outlines the detailed architecture and implementation steps for building the backend of a stock advisory system. The system uses AI agents (powered by Gemini via LangChain) to handle user queries about stock data, technical analysis, predictions, and recommendations. It integrates databases (PostgreSQL for structured data, Qdrant for vector search on financial news), machine learning tools for predictions, and external APIs (e.g., VNStock).
 
The goal of this plan is to provide a structured blueprint so that Grok AI can generate code for each component. You can copy-paste sections of this plan into prompts for Grok AI, e.g., "Generate Python code for the [specific section] based on this description: [paste details]".
 
**Key Technologies:**
- Backend Framework: FastAPI (for API endpoints).
- AI Orchestration: LangChain (for agents, tools, and chains).
- LLM: Google Gemini (2.0 Flash for planning, Pro for responses).
- Databases: PostgreSQL (for stock data, financial reports, conversation history).
- Vector DB: Qdrant (for semantic search on financial news embeddings).
- ML: Pre-trained models in core_services (e.g., for prediction, indicators).
- Other Libraries: sentence-transformers (for embeddings), tenacity (for retries), psycopg2 (for Postgres), TA-Lib (for technical indicators), matplotlib/plotly (for visualizations).
 
**Assumptions:**
- This is a student project; no advanced security (e.g., JWT) or scaling (e.g., Docker).
- Frontend (FE) calls the backend API via POST /chat with query and session_id.
- Tools are defined using LangChain's @tool decorator.
- ML models are already trained and exposed as internal functions/endpoints in core_services.
- News crawling tool exists and feeds data to Qdrant (not implemented here; assume it's a separate script).
 
**Project Structure:**
```
stock-advisory-backend/
├── app.py                # Main FastAPI app
├── agents.py             # Define Master, Plan, Response agents
├── tools.py              # Define all tools (DB queries, ML, search, etc.)
├── db.py                 # Postgres connection and history management
├── qdrant_setup.py       # Qdrant client and indexing functions
├── core_services/        # ML models, indicators, VNStock API wrappers
│   ├── predict.py
│   ├── indicators.py
│   ├── vnstock.py
│   └── visualize.py
├── requirements.txt      # Dependencies
└── plan.md               # This file
```
 
## Step 1: Setup Environment
- Install dependencies: Create `requirements.txt` with:
  ```
  fastapi
  uvicorn
  langchain
  langchain-google-genai
  google-generativeai
  psycopg2-binary
  qdrant-client
  sentence-transformers
  tenacity
  ta-lib  # For indicators
  matplotlib  # For visualizations
  plotly
  pandas  # For data handling
  ```
- Prompt for Grok AI: "Generate a requirements.txt file and a basic setup script to install dependencies for this project."
 
## Step 2: Database Setup (PostgreSQL)
- Use Postgres for:
  - Stock data (prices, financial reports - BCTC).
  - Conversation history (for state management).
- Schema:
  - Table `stock_prices`: columns (stock_code: str, date: date, open: float, close: float, etc.).
  - Table `financial_reports`: columns (company: str, quarter/year: str, revenue: float, etc.).
  - Table `conversation_history`: columns (id: serial PK, session_id: varchar, query: text, response: text, timestamp: timestamp).
- Implement in `db.py`:
  - Connection function.
  - `get_history_summary(session_id: str, max_items: int = 5) -> str`: Query last 5 queries/responses, format as string.
  - `save_to_history(session_id: str, query: str, response: str)`: Insert new entry.
- Prompt for Grok AI: "Generate Python code for db.py with PostgreSQL connection, schema creation (if not exists), and functions for conversation history management. Use psycopg2."
 
## Step 3: Qdrant Setup for News Search
- Use Qdrant for vector search on financial news.
- Collection: "financial_news" with vector size 384 (from sentence-transformers 'all-MiniLM-L6-v2').
- Implement in `qdrant_setup.py`:
  - Client connection (local: localhost:6333).
  - `index_news(articles: list[dict])`: Embed texts and upsert to Qdrant. Article format: {'id': int, 'text': str, 'metadata': {'stock': str}}.
  - Assume a separate crawler script calls this to index news periodically.
- Prompt for Grok AI: "Generate Python code for qdrant_setup.py to connect to Qdrant, create collection if needed, and index news articles using sentence-transformers."
 
## Step 4: Define Tools (tools.py)
- All tools use LangChain's @tool decorator.
- List of tools:
  1. `get_from_postgres(query_type: str, params: dict) -> dict`: Handle queries like get prices, financial data. Use SQL based on query_type (e.g., 'prices', 'reports').
  2. `qdrant_search_news(query: str, stock_code: str = None, top_k: int = 5) -> list[dict]`: Semantic search for news.
  3. `vnstock_api_tool(company: str) -> dict`: Call VNStock API for business results.
  4. `indicator_tool(stock_code: str, indicator: str, period: str) -> dict`: Compute RSI, MA, etc., using TA-Lib on data from DB.
  5. `predict_tool(stock_code: str, data: dict) -> dict`: Call ML model for trend prediction or buy/sell points.
  6. `visualize_tool(data: dict, viz_type: str) -> str`: Generate chart (line, bar) as base64 image using matplotlib/plotly.
  7. `portfolio_allocation(stocks: list, horizon: str, capital: float) -> dict`: For capital allocation (short/long term) using simple MPT logic.
- Each tool should have docstring for LangChain to use in planning.
- Add retry with tenacity to error-prone tools (e.g., API calls).
- Prompt for Grok AI: "Generate Python code for tools.py defining all the tools listed above using LangChain @tool. Include examples for each."
 
## Step 5: Define Agents (agents.py)
- Use LangChain for agents.
- Plan Agent: ZeroShotReactDescriptionAgent with Gemini Flash, prompt to output JSON plan {'tools': list, 'complete': bool}.
- Response Agent: Simple LLMChain with Gemini Pro to generate user-friendly response (text + charts).
- Master Orchestration: AgentExecutor with max_iterations=3.
- Prompt template for Plan Agent: Include history and previous data.
- Prompt for Grok AI: "Generate Python code for agents.py defining Plan Agent, Response Agent, and the AgentExecutor setup using LangChain and Gemini."
 
## Step 6: Main API (app.py)
- FastAPI app with single endpoint: POST /chat {query: str, session_id: str}.
- Flow:
  1. Get history summary from DB.
  2. Run AgentExecutor (Plan Agent loop).
  3. Call Response Agent with results.
  4. Save to history.
  5. Return JSON response {response: str} (or with base64 charts).
- Include error handling with fallback.
- Run with uvicorn.
- Prompt for Grok AI: "Generate the full app.py for FastAPI backend integrating db.py, tools.py, agents.py, and qdrant_setup.py. Include the /chat endpoint logic."
 
## Step 7: Core Services (core_services/)
- Separate modules for ML and utilities.
- predict.py: Functions for stock prediction (e.g., dummy LSTM-like logic if no real model).
- indicators.py: TA-Lib wrappers.
- vnstock.py: API client for VNStock.
- visualize.py: Chart generation functions.
- Prompt for Grok AI: "Generate code for core_services/predict.py implementing a simple ML prediction function for stock trends."
 
## Step 8: Testing and Running
- Test plan: Unit tests for tools (use pytest), integration test for full flow.
- Run: `uvicorn app:app --reload`.
- Add a script to seed DB with sample data.
- Prompt for Grok AI: "Generate a test script and sample data seeder for the database."
 
## Step 9: Handling Specific Use Cases
- Query Data: Tools get_from_postgres + visualize.
- Technical Analysis: indicator_tool + visualize.
- Prediction: predict_tool.
- Recommendations: Combine tools + Response Agent for synthesis.
- News Integration: Use qdrant_search_news in plans for contextual info.
 
## Usage with Grok AI for Code Gen
- For each section, prompt: "Based on this plan section [paste section], generate the corresponding Python code file."
- Iterate: After gen one file, test it, then gen next.
- If issues: Provide feedback in next prompt, e.g., "Fix this code: [paste code] with error [describe]."
 
This plan is comprehensive; start with setup and build incrementally. If you need expansions, update this MD!