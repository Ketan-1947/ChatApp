# RAG Chatbot with Conversation Memory

A full-screen dark mode chatbot that answers questions about uploaded PDF documents using Retrieval-Augmented Generation (RAG) with conversation memory. Built with FastAPI backend and modern web interface.

## Description

This project combines a modern web interface with a machine learning backend to create an intelligent question-answering system for PDF documents. The system can:

- **Upload and process PDF documents** for content analysis
- **Answer questions** about document content using RAG
- **Maintain conversation memory** (up to 10 previous exchanges)
- **Provide general knowledge answers** when PDF context isn't available
- **Generate document summaries** without conversation influence
- **Handle follow-up questions** with context awareness

## Architecture

- **Backend**: FastAPI (Python) - `app.py`
- **RAG Model**: Custom implementation with Groq LLM - `model.py`
- **Embeddings**: HuggingFace sentence-transformers
- **Vector Store**: FAISS for similarity search
- **Frontend**: Static HTML/CSS/JavaScript (dark mode)

## Features

### Core Functionality
- 📄 **PDF Processing**: Upload and analyze PDF documents
- 🤖 **Intelligent QA**: Context-aware question answering
- 🧠 **Memory Management**: Remembers last 10 conversations
- 📝 **Document Summarization**: Generate comprehensive summaries
- 🔄 **Fallback System**: General knowledge when PDF context insufficient

### API Endpoints
- `POST /api/upload` - Upload PDF documents
- `POST /api/chat` - Main chat interface with intelligent routing
- `POST /api/summarize-pdf` - Generate document summaries
- `POST /api/summarize-text` - Summarize provided text
- `POST /api/general-qa` - General knowledge questions
- `DELETE /api/reset` - Clear session and memory
- `DELETE /api/clear-memory` - Clear conversation memory only
- `GET /api/status` - System status information

## Setup

### Prerequisites
- Python 3.8+
- Groq API key
- spaCy English model

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rag-chatbot
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download spaCy model**:
```bash
python -m spacy download en_core_web_sm
```

4. **Set up environment variables**:
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

5. **Run the application**:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

6. **Open your browser**:
```
http://localhost:8000
```

## Usage

### Basic Workflow
1. **Upload a PDF** using the paperclip icon or drag-and-drop
2. **Wait for processing** (document will be indexed)
3. **Ask questions** about the document content
4. **Get contextual answers** based on document and conversation history

### Example Interactions
```
User: "What is the main topic of this document?"
Bot: [Analyzes PDF content and provides answer]

User: "Can you elaborate on that?"
Bot: [Understands "that" refers to previous answer due to memory]

User: "Summarize the document"
Bot: [Generates fresh summary without conversation bias]
```

### Memory Management
- **Automatic**: Stores last 10 conversations automatically
- **Manual**: Use `/api/clear-memory` to reset conversation history
- **Persistent**: Memory maintained during session
- **Contextual**: Follow-up questions understand previous exchanges

## Configuration

### Conversation Memory
```python
# In model.py initialization
rag = RAG(max_conversation_history=10)  # Adjust memory size

# Runtime adjustment
rag.set_max_conversation_history(15)
```

### Model Parameters
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `meta-llama/llama-4-scout-17b-16e-instruct` (Groq)
- **Text Chunking**: 128 tokens with 32 overlap
- **Vector Search**: Top 5 similar documents

## File Structure

```
├── app.py              # FastAPI application
├── model.py            # RAG model with conversation memory
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── uploads/           # Uploaded PDF storage
└── static/            # Frontend files
    ├── index.html
    ├── style.css
    └── script.js
```
