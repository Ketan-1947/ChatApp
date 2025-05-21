from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
from typing import Optional
from pydantic import BaseModel
from model import RAG

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

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

# Initialize the RAG model
# You might want to adjust these model names based on your specific needs
rag_model = RAG(
    sent_model_name="sentence-transformers/all-MiniLM-L6-v2",
    generator_model_name="google/flan-t5-base"
)

# Global variable to track if a document has been loaded
document_loaded = False

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatResponse(BaseModel):
    reply: str
    source: str  # Indicates the source of the answer: "pdf", "general", or "summary"

class SummaryResponse(BaseModel):
    summary: str

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file for RAG model to process.
    """
    global document_loaded
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = f"uploads/{file.filename}"
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process the PDF with the RAG model
        rag_model.load_pdf(file_path)
        document_loaded = True
        return {"message": f"Successfully loaded {file.filename}"}
    except Exception as e:
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
    
    # Handle file upload if provided
    if file:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_path = f"uploads/{file.filename}"
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Process the PDF with the RAG model
            rag_model.load_pdf(file_path)
            document_loaded = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    try:
        # Check if the message is a request for summarization
        if "summarize" in message.lower():
            if not document_loaded:
                return ChatResponse(
                    reply="Please upload a PDF document first so I can summarize it.",
                    source="general"
                )
            summary = rag_model.summarize_pdf()
            return ChatResponse(reply=summary, source="summary")
        
        # If PDF is loaded, try to answer from PDF first
        if document_loaded:
            pdf_answer = rag_model.get_answer(message)
            
            # If the answer seems too generic or short, use general QA
            if len(pdf_answer.split()) < 10 or "sorry" in pdf_answer.lower():
                general_answer = rag_model.answer_general_question(message)
                return ChatResponse(reply=general_answer, source="general")
            
            return ChatResponse(reply=pdf_answer, source="pdf")
        
        # If no PDF is loaded, use general QA
        general_answer = rag_model.answer_general_question(message)
        return ChatResponse(reply=general_answer, source="general")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/api/summarize-text", response_model=SummaryResponse)
async def summarize_text(text: str = Form(...)):
    """
    Summarize a given text.
    """
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
    
    if file:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_path = f"uploads/{file.filename}"
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            summary = rag_model.summarize_pdf(file_path)
            document_loaded = True
            return SummaryResponse(summary=summary)
        except Exception as e:
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
    try:
        answer = rag_model.answer_general_question(question)
        return ChatResponse(reply=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
async def redirect_to_index():
    """
    Redirect root endpoint to the static index.html
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)