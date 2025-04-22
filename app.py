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
    Process a chat message and optionally upload a file.
    """
    global document_loaded
    
    # If a file is uploaded along with the message
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
    
    # If no document has been loaded yet
    if not document_loaded:
        return ChatResponse(reply="Please upload a PDF document first so I can answer questions about it.")
    
    try:
        # Generate answer using the RAG model
        answer = rag_model.get_answer(message)
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