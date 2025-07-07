from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
from typing import Optional
from pydantic import BaseModel
from model import RAG
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API with Groq")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Initialize the RAG model with Groq
try:
    rag_model = RAG(sent_model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("RAG model initialized successfully with Groq")
except Exception as e:
    print(f"Error initializing RAG model: {e}")
    rag_model = None

# Global variable to track if a document has been loaded
document_loaded = False

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatResponse(BaseModel):
    reply: str
    source: str  # Indicates the source of the answer: "pdf", "general", or "summary"
    conversation_length: int  # Number of exchanges in memory

class SummaryResponse(BaseModel):
    summary: str

class StatusResponse(BaseModel):
    status: str
    message: str

class MemoryResponse(BaseModel):
    message: str
    conversation_length: int

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return StatusResponse(
        status="healthy" if rag_model else "unhealthy",
        message="RAG model is ready" if rag_model else "RAG model not initialized"
    )

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file for RAG model to process.
    """
    global document_loaded
    
    if not rag_model:
        raise HTTPException(status_code=503, detail="RAG model not initialized")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = f"uploads/{file.filename}"
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF with the RAG model
        rag_model.load_pdf(file_path)
        document_loaded = True
        
        return {"message": f"Successfully loaded {file.filename}", "status": "success"}
    except Exception as e:
        # Clean up the file if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Intelligent chat endpoint that handles all types of requests:
    - PDF summarization
    - PDF-based QA
    - General QA
    """
    global document_loaded
    
    if not rag_model:
        raise HTTPException(status_code=503, detail="RAG model not initialized")
    
    # Handle file upload if provided
    if file:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_path = f"uploads/{file.filename}"
        
        try:
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process the PDF with the RAG model
            rag_model.load_pdf(file_path)
            document_loaded = True
        except Exception as e:
            # Clean up the file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    try:
        # Check if the message is a request for summarization
        summarize_keywords = ["summarize", "summary", "sum up", "brief", "overview"]
        if any(keyword in message.lower() for keyword in summarize_keywords):
            if not document_loaded:
                return ChatResponse(
                    reply="Please upload a PDF document first so I can summarize it.",
                    source="general"
                )
            summary = rag_model.summarize_pdf()
            return ChatResponse(reply=summary, source="summary", conversation_length=0)
        
        # If PDF is loaded, try to answer from PDF first
        if document_loaded:
            pdf_answer = rag_model.get_answer(message)
            
            # If the answer seems too generic or indicates no specific answer found
            if (len(pdf_answer.split()) < 15 or 
                "cannot find a specific answer" in pdf_answer.lower() or
                "i don't know" in pdf_answer.lower() or
                "i'm not sure" in pdf_answer.lower()):
                
                # Try general QA as fallback
                general_answer = rag_model.answer_general_question(message)
                return ChatResponse(
                    reply=general_answer, 
                    source="general",
                    conversation_length=len(rag_model.conversation_history)
                )
            
            return ChatResponse(
                reply=pdf_answer, 
                source="pdf",
                conversation_length=len(rag_model.conversation_history)
            )
        
        # If no PDF is loaded, use general QA
        general_answer = rag_model.answer_general_question(message)
        return ChatResponse(
            reply=general_answer, 
            source="general",
            conversation_length=len(rag_model.conversation_history)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/api/summarize-text", response_model=SummaryResponse)
async def summarize_text(text: str = Form(...)):
    """
    Summarize a given text.
    """
    if not rag_model:
        raise HTTPException(status_code=503, detail="RAG model not initialized")
    
    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text must be at least 10 characters long")
    
    try:
        summary = rag_model.summarize_text(text)
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.post("/api/summarize-pdf", response_model=SummaryResponse)
async def summarize_pdf(file: Optional[UploadFile] = File(None)):
    """
    Summarize the currently loaded PDF or a newly uploaded PDF.
    """
    global document_loaded
    
    if not rag_model:
        raise HTTPException(status_code=503, detail="RAG model not initialized")
    
    if file:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_path = f"uploads/{file.filename}"
        
        try:
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process and summarize the PDF
            summary = rag_model.summarize_pdf(file_path)
            document_loaded = True
            return SummaryResponse(summary=summary)
        except Exception as e:
            # Clean up the file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    if not document_loaded:
        raise HTTPException(status_code=400, detail="No PDF document has been loaded")
    
    try:
        summary = rag_model.summarize_pdf()
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.post("/api/general-qa", response_model=ChatResponse)
async def general_qa(question: str = Form(...)):
    """
    Answer a general question without requiring PDF context.
    """
    if not rag_model:
        raise HTTPException(status_code=503, detail="RAG model not initialized")
    
    if not question or len(question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question must be at least 3 characters long")
    
    try:
        answer = rag_model.answer_general_question(question)
        return ChatResponse(
            reply=answer, 
            source="general",
            conversation_length=len(rag_model.conversation_history)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.delete("/api/reset")
async def reset_session():
    """Reset the current session by clearing the loaded document and conversation memory."""
    global document_loaded
    document_loaded = False
    
    # Clear the vector store and conversation memory
    if rag_model:
        rag_model.vectorstore = None
        rag_model.clear_memory()
    
    # Clean up uploaded files (optional)
    try:
        for filename in os.listdir("uploads"):
            file_path = os.path.join("uploads", filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not clean up uploads directory: {e}")
    
    return {"message": "Session and conversation memory reset successfully", "status": "success"}

@app.delete("/api/clear-memory", response_model=MemoryResponse)
async def clear_memory():
    """Clear only the conversation memory, keeping the loaded document."""
    if not rag_model:
        raise HTTPException(status_code=503, detail="RAG model not initialized")
    
    rag_model.clear_memory()
    return MemoryResponse(
        message="Conversation memory cleared successfully",
        conversation_length=0
    )

@app.get("/api/memory", response_model=MemoryResponse)
async def get_memory_status():
    """Get the current conversation memory status."""
    if not rag_model:
        raise HTTPException(status_code=503, detail="RAG model not initialized")
    
    return MemoryResponse(
        message=f"Conversation has {len(rag_model.conversation_history)} exchanges in memory",
        conversation_length=len(rag_model.conversation_history)
    )

@app.get("/api/status")
async def get_status():
    """Get the current status of the RAG system."""
    return {
        "model_initialized": rag_model is not None,
        "document_loaded": document_loaded,
        "conversation_length": len(rag_model.conversation_history) if rag_model else 0,
        "uploads_directory": os.path.exists("uploads"),
        "groq_model": "meta-llama/llama-4-scout-17b-16e-instruct"
    }

@app.get("/")
async def redirect_to_index():
    """
    Redirect root endpoint to the static index.html
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)