# LegalSmart Chatbot

<p align="center">
  <img src="chatbot_legal.jpeg" alt="LegalSmart Chatbot banner" width="900"/>
</p>

LegalSmart Chatbot is a Retrieval-Augmented Generation (RAG) legal assistant focused on Indian law. It answers user questions using context retrieved from the **Indian Penal Code (IPC)**, **Bharatiya Nyaya Sanhita (BNS)**, and the **Constitution of India**.

The project is implemented with **Streamlit**, **LangChain**, **FAISS**, **Hugging Face embeddings**, and **Together AI-hosted LLMs**.

## Research Publication

This project is part of our published research work:

- **Springer (book chapter):** https://link.springer.com/chapter/10.1007/978-3-032-12827-0_7

## Repository Description (for GitHub)

> A RAG-based legal assistant for Indian law that uses FAISS retrieval over IPC, BNS, and Constitution documents to generate grounded answers via Streamlit + LangChain.

## Suggested GitHub Topics

Use these topics in your repository settings to improve discoverability:

- `rag`
- `legal-ai`
- `chatbot`
- `streamlit`
- `langchain`
- `faiss`
- `llm`
- `retrieval-augmented-generation`
- `indian-law`
- `together-ai`
- `huggingface`
- `python`

## Features

- Grounded legal Q&A over IPC, BNS, and Constitutional text.
- Conversational memory for multi-turn interactions.
- Retrieval pipeline powered by FAISS vector search.
- Streamlit-based chat interface for easy deployment.
- Prompting strategy designed for both non-legal and legal-professional audiences.

## Tech Stack

- **App/UI:** Streamlit
- **LLM Orchestration:** LangChain
- **Vector Store:** FAISS
- **Embeddings:** `nomic-ai/nomic-embed-text-v1` (Hugging Face)
- **LLM Provider:** Together AI

## Project Structure

```text
.
├── app.py                  # Streamlit chat app
├── Ingest.py               # Document ingestion + vector indexing
├── requirements.txt        # Python dependencies
├── data/                   # Source legal PDFs (IPC, BNS, Constitution)
└── law_vector_db/          # Generated FAISS index files
```

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/srujankothuri/LegalSmart-Chatbot.git
cd LegalSmart-Chatbot
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Build the vector database

Run ingestion first (recommended on a machine with sufficient RAM/CPU):

```bash
python Ingest.py
```

This generates/updates the `law_vector_db` directory used at runtime.

### 4) Configure API key

Set your Together AI API key as an environment variable:

```bash
export TOGETHER_API_KEY="your_together_api_key"
```

> If deploying on Streamlit Cloud/Hugging Face Spaces, store `TOGETHER_API_KEY` in the platform's secrets settings.

### 5) Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`).

## Notes

- This tool is for educational and informational purposes.
- Responses may not be a substitute for professional legal advice.
- Always verify critical legal decisions with a qualified legal practitioner.

## Contributing

Issues and pull requests are welcome. If you find bugs or want improvements, please open an issue first to discuss scope.

## Contact

For feedback or questions, open an issue:

- https://github.com/srujankothuri/LegalSmart-Chatbot/issues
