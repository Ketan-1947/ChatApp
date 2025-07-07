import os
# Set environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from groq import Groq
import torch
import logging
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from collections import deque
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, sent_model_name="sentence-transformers/all-MiniLM-L6-v2", max_conversation_history=10):
        """Initialize the RAG model with specified embedding model and Groq client."""
        try:
            logger.info(f"Initializing RAG with embedding model: {sent_model_name}")
            
            # Initialize Groq client with API key from environment
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
            
            self.groq_client = Groq(api_key=groq_api_key)
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=sent_model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            # Initialize text splitter
            logger.info("Initializing text splitter")
            self.text_splitter = SpacyTextSplitter(
                pipeline="en_core_web_sm",
                chunk_size=128,
                chunk_overlap=32
            )
            
            # Initialize vector store
            self.vectorstore = None
            
            # Initialize conversation history
            self.max_conversation_history = max_conversation_history
            self.conversation_history = deque(maxlen=max_conversation_history)
            
            logger.info("RAG model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG model: {str(e)}")
            raise

    def load_pdf(self, pdf_path):
        """Load a PDF document and create embeddings for its content."""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            logger.info(f"Loading PDF: {pdf_path}")
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split documents into chunks
            logger.info("Splitting documents into chunks")
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(texts)} text chunks")
            
            # Create vector store
            logger.info("Creating vector store")
            self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            logger.info(f"Vector store created with {len(texts)} documents")
            
            return True
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise

    def _add_to_conversation_history(self, user_message, assistant_response):
        """Add a conversation turn to the history."""
        conversation_turn = {
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": assistant_response
        }
        self.conversation_history.append(conversation_turn)
        logger.info(f"Added conversation turn to history. Total history: {len(self.conversation_history)}")

    def _get_conversation_context(self):
        """Get formatted conversation history for context."""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for i, turn in enumerate(self.conversation_history):
            context_parts.append(f"Previous conversation {i+1}:")
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['assistant']}")
            context_parts.append("")  # Empty line for readability
        
        return "\n".join(context_parts)

    def _get_groq_response(self, messages, temperature=0.7, max_tokens=1024):
        """Get response from Groq API with streaming."""
        try:
            completion = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=1,
                stream=True,
                stop=None,
            )
            
            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            return response_text.strip()
        except Exception as e:
            logger.error(f"Error getting Groq response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now."

    def get_answer(self, question, k=5):
        """Generate an answer to a question using the RAG approach with conversation history."""
        try:
            if not self.vectorstore:
                answer = "I cannot find a specific answer in the document."
                self._add_to_conversation_history(question, answer)
                return answer
            
            logger.info(f"Generating answer for question: {question}")
            
            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(question, k=k)
            
            if not docs:
                answer = "I cannot find a specific answer in the document."
                self._add_to_conversation_history(question, answer)
                return answer
            
            # Combine context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Get conversation history context
            conversation_context = self._get_conversation_context()
            
            # Create messages for Groq API with conversation history
            system_message = """You are a helpful assistant that answers questions based on provided context. 
You have access to conversation history to maintain context and provide better responses.
If you cannot find the answer in the context, say "I cannot find a specific answer in the document."
Use the conversation history to understand follow-up questions and maintain continuity."""
            
            user_content = f"""Previous conversation history:
{conversation_context}

Current document context: {context}

Current question: {question}

Answer:"""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
            
            # Get answer from Groq
            answer = self._get_groq_response(messages)
            
            # Check answer quality
            if self._is_low_quality_answer(answer):
                answer = "I cannot find a specific answer in the document."
            
            # Add to conversation history
            self._add_to_conversation_history(question, answer)
            
            logger.info("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            error_response = "I cannot find a specific answer in the document."
            self._add_to_conversation_history(question, error_response)
            return error_response

    def summarize_text(self, text, max_length=512):
        """Summarize a given text using the Groq API with conversation history."""
        try:
            logger.info("Generating summary for text")
            
            # Get conversation history context
            conversation_context = self._get_conversation_context()
            
            # Create messages for Groq API with conversation history
            system_message = """You are a helpful assistant that summarizes text. 
You have access to conversation history to maintain context and provide better responses.
Consider the conversation history when generating summaries."""
            
            user_content = f"""Previous conversation history:
{conversation_context}

Provide a comprehensive summary of the following text. 
Include the main topics, key points, and important details.
If the text is too long, focus on the most important information.

Text: {text}

Summary:"""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
            
            # Generate summary using Groq
            summary = self._get_groq_response(messages, temperature=0.5)
            
            # Add to conversation history
            self._add_to_conversation_history(f"Summarize: {text[:100]}...", summary)
            
            logger.info("Summary generated successfully")
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            error_response = f"Sorry, I encountered an error while generating the summary: {str(e)}"
            self._add_to_conversation_history(f"Summarize: {text[:100]}...", error_response)
            return error_response

    def answer_general_question(self, question):
        """Answer a general question without requiring PDF context but with conversation history."""
        try:
            logger.info(f"Answering general question: {question}")
            
            # Get conversation history context
            conversation_context = self._get_conversation_context()
            
            # Create messages for Groq API with conversation history
            system_message = """You are a helpful assistant that answers questions in a clear and informative way. 
You have access to conversation history to maintain context and provide better responses.
If you're not sure about something, say so. Use the conversation history to understand follow-up questions."""
            
            user_content = f"""Previous conversation history:
{conversation_context}

Current question: {question}

Answer:"""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
            
            # Generate answer using Groq
            answer = self._get_groq_response(messages)
            
            # Add to conversation history
            self._add_to_conversation_history(question, answer)
            
            logger.info("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            error_response = "I apologize, but I'm having trouble generating an answer right now."
            self._add_to_conversation_history(question, error_response)
            return error_response

    def summarize_pdf(self, pdf_path=None, max_chunks=10):
        """Summarize the content of a PDF document without using conversation history."""
        try:
            # If no PDF path is provided, use the currently loaded PDF
            if pdf_path:
                self.load_pdf(pdf_path)
            
            if not self.vectorstore:
                answer = "No document has been loaded for summarization."
                self._add_to_conversation_history("Summarize PDF", answer)
                return answer
            
            logger.info("Generating PDF summary")
            
            # Get all documents from the vector store
            all_docs = self.vectorstore.similarity_search("", k=max_chunks)
            
            if not all_docs:
                answer = "No content found in the document."
                self._add_to_conversation_history("Summarize PDF", answer)
                return answer
            
            # Combine text from documents
            full_text = " ".join([doc.page_content for doc in all_docs])
            
            # Create messages for Groq API without conversation history
            messages = [
                {
                    "role": "user",
                    "content": f"""Provide a comprehensive summary of the following document. 
Include the main topics, key points, and important details.
If the text is too long, focus on the most important information.

Document content: {full_text}

Summary:"""
                }
            ]
            
            # Generate summary using Groq
            summary = self._get_groq_response(messages, temperature=0.5, max_tokens=1024)
            
            # Add to conversation history
            self._add_to_conversation_history("Summarize PDF", summary)
            
            logger.info("PDF summary generated successfully")
            return summary
        except Exception as e:
            logger.error(f"Error generating PDF summary: {str(e)}")
            error_response = "I apologize, but I'm having trouble generating a summary right now."
            self._add_to_conversation_history("Summarize PDF", error_response)
            return error_response

    def _is_low_quality_answer(self, answer):
        """Check if an answer is of low quality."""
        # Check for short answers
        if len(answer.split()) < 10:
            return True
            
        # Check for generic responses
        generic_phrases = [
            "i cannot find",
            "i don't know",
            "i'm not sure",
            "i apologize",
            "i'm having trouble",
            "no specific answer",
            "cannot find a specific answer"
        ]
        
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in generic_phrases)

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def get_conversation_history(self):
        """Get the current conversation history."""
        return list(self.conversation_history)

    def set_max_conversation_history(self, max_history):
        """Set the maximum number of conversations to keep in history."""
        self.max_conversation_history = max_history
        # Create new deque with updated maxlen
        current_history = list(self.conversation_history)
        self.conversation_history = deque(current_history, maxlen=max_history)
        logger.info(f"Maximum conversation history set to {max_history}")

    def export_conversation_history(self, file_path):
        """Export conversation history to a file."""
        try:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(list(self.conversation_history), f, indent=2, ensure_ascii=False)
            logger.info(f"Conversation history exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting conversation history: {str(e)}")
            return False

    def import_conversation_history(self, file_path):
        """Import conversation history from a file."""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            self.conversation_history = deque(history_data, maxlen=self.max_conversation_history)
            logger.info(f"Conversation history imported from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error importing conversation history: {str(e)}")
            return False