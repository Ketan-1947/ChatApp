# RAG Chatbot

A full-screen dark mode chatbot that answers questions about uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## Description

This project combines a modern web interface with a machine learning backend to create a question-answering system for PDF documents. Upload any PDF and ask questions about its content.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application with uvicorn reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. Open your browser and go to:
```
http://localhost:8000
```

## Usage

1. Upload a PDF using the paperclip icon
2. Wait for the document to process
3. Ask questions about the document content