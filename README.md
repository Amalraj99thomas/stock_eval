# •	Stock Evaluation RAG System
Company Stock Evaluation API (FastAPI + RAG)
Designed a stock evaluating RAG application with FastAPI framework. The service asks for a company name, then searches high quality finance sources for current stock evaluation, recent news and trading sentiments. This information is scraped, cleaned and processing to store as embeddings in ChromaDB vector store for each target company. A RAG based chatbot is designed to answer stock related questions to assist investment and trading.  
