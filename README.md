# OVGU / FIN / Magdeburg Assistant

A conversational AI assistant providing information about Otto von Guericke University Magdeburg (OVGU), the Faculty of Informatics (FIN), and the city of Magdeburg. Built with Streamlit, LangGraph, and Supabase.

## Features

*   **Conversational Interface:** Ask questions in natural language via a Streamlit web UI.
*   **Multi-Agent System:** Uses LangGraph to route queries to specialized agents (OVGU, FIN, Magdeburg - inferred).
*   **Information Retrieval:** (Likely) retrieves information from ingested data sources, potentially stored in Supabase.
*   **LLM Integration:** Leverages Large Language Models (like OpenAI's GPT) for understanding queries and generating responses.
*   **Source Linking:** Formats responses to include clickable source links where available.

## Tech Stack

*   **Frontend:** Streamlit
*   **Backend/Orchestration:** Python, LangGraph
*   **LLM:** OpenAI (configurable via `.env`)
*   **Database:** Supabase (optional, based on setup)
*   **Data Ingestion:** Crawl4AI (likely used in `ingestion/`)
*   **Dependencies:** See `requirements.txt`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    *   Copy the example environment file: `cp .env.example .env` (or `copy .env.example .env` on Windows).
    *   Edit the `.env` file and add your credentials (see Environment Variables section below).

## Environment Variables

Create a `.env` file in the project root and add the following variables:

```dotenv
# Required if using OpenAI models
OPENAI_API_KEY=your_openai_api_key

# Required if using the Supabase integration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_role_key

# Optional: Specify the OpenAI model to use (defaults might exist in code)
# Example: gpt-4o-mini, gpt-4, etc.
LLM_MODEL=gpt-4o-mini
```

*   Get OpenAI keys from [platform.openai.com](https://platform.openai.com/).
*   Get Supabase URL and Service Key from your Supabase project settings under API.

## Usage

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Open your web browser to the URL provided by Streamlit (usually `http://localhost:8501`). Interact with the assistant by typing questions into the chat input.

## Project Structure

```
.
├── .env                    # Local environment variables (ignored by git)
├── .env.example            # Example environment variables file
├── .gitignore              # Files ignored by Git
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── streamlit_app.py        # Main Streamlit application
├── supabase_schema_separate.sql # SQL schema for Supabase tables
├── utils.py                # Utility functions
├── agents/                 # Code for different agents (e.g., OVGU, FIN, Magdeburg)
├── data/                   # Directory for storing data (e.g., crawled content)
├── graph/                  # LangGraph agent graph definition
├── ingestion/              # Scripts for data ingestion (e.g., web crawling)
└── venv/                   # Virtual environment directory (ignored by git)
``` 