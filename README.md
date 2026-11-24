# Telecom Customer Support AI Assistant

An AI-powered telecom support assistant that uses Hybrid RAG, SQL RAG, and Semantic Routing to answer user queries from documents and structured support ticket data.

### Features
- **Semantic Routing:** Routes user questions to the appropriate RAG or SQL workflow.
- **Hybrid RAG:** Retrieves relevant information from FAQs and PDF documents using dense and sparse search.
- **SQL RAG:** Converts natural-language questions into SQL queries to retrieve support ticket information.
- **Document Processing:** Uses Docling and HybridChunker for document conversion and chunking.
- **API:** Exposes the application using FastAPI.

### Tech Stack
Python, LangChain, Semantic Router, Docling, Qdrant, Hugging Face, Groq, SQLite, FastAPI.
