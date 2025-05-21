import os
# Set environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import logging
from tqdm import tqdm
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, sent_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 generator_model_name="google/flan-t5-base"):
        """Initialize the RAG model with specified embedding and generator models."""
        try:
            logger.info(f"Initializing RAG with embedding model: {sent_model_name}")
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
            
            # Initialize generator model
            logger.info(f"Initializing generator model: {generator_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
            
            # Create pipeline
            pipe = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize LLM
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            # Initialize vector store
            self.vectorstore = None
            
            # Initialize prompt templates
            self.pdf_qa_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""Use the following context to answer the question. 
                If you cannot find the answer in the context, say "I cannot find a specific answer in the document."
                Context: {context}
                Question: {question}
                Answer:"""
            )
            
            self.general_qa_template = PromptTemplate(
                input_variables=["question"],
                template="""Answer the following question in a clear and informative way. 
                If you're not sure about something, say so.
                Question: {question}
                Answer:"""
            )
            
            self.pdf_summary_template = PromptTemplate(
                input_variables=["text"],
                template="""Provide a comprehensive summary of the following document. 
                Include the main topics, key points, and important details.
                If the text is too long, focus on the most important information.
                Text: {text}
                Summary:"""
            )
            
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

    def get_answer(self, question, k=5):
        """Generate an answer to a question using the RAG approach."""
        try:
            if not self.vectorstore:
                return "I cannot find a specific answer in the document."
            
            logger.info(f"Generating answer for question: {question}")
            
            # Create QA chain with custom prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.pdf_qa_template}
            )
            
            # Get answer
            result = qa_chain({"query": question})
            answer = result["result"]
            
            # Check answer quality
            if self._is_low_quality_answer(answer):
                return "I cannot find a specific answer in the document."
            
            logger.info("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I cannot find a specific answer in the document."

    def summarize_text(self, text, max_length=512):
        """Summarize a given text using the language model."""
        try:
            logger.info("Generating summary for text")
            
            # Create summarization chain
            summarize_chain = LLMChain(
                llm=self.llm,
                prompt=self.summarize_template
            )
            
            # Generate summary
            summary = summarize_chain.run(text=text)
            
            logger.info("Summary generated successfully")
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Sorry, I encountered an error while generating the summary: {str(e)}"

    def answer_general_question(self, question):
        """Answer a general question without requiring PDF context."""
        try:
            logger.info(f"Answering general question: {question}")
            
            # Create general QA chain
            qa_chain = LLMChain(
                llm=self.llm,
                prompt=self.general_qa_template
            )
            
            # Generate answer
            answer = qa_chain.run(question=question)
            
            logger.info("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I apologize, but I'm having trouble generating an answer right now."

    def summarize_pdf(self, pdf_path=None, max_chunks=10):
        """Summarize the content of a PDF document."""
        try:
            # If no PDF path is provided, use the currently loaded PDF
            if pdf_path:
                self.load_pdf(pdf_path)
            
            if not self.vectorstore:
                return "No document has been loaded for summarization."
            
            logger.info("Generating PDF summary")
            
            # Get all documents from the vector store
            documents = self.vectorstore.get()
            if not documents:
                return "No content found in the document."
            
            # Combine text from documents
            full_text = " ".join([doc.page_content for doc in documents[:max_chunks]])
            
            # Create PDF summary chain
            pdf_summary_chain = LLMChain(
                llm=self.llm,
                prompt=self.pdf_summary_template
            )
            
            # Generate summary
            summary = pdf_summary_chain.run(text=full_text)
            
            logger.info("PDF summary generated successfully")
            return summary
        except Exception as e:
            logger.error(f"Error generating PDF summary: {str(e)}")
            return "I apologize, but I'm having trouble generating a summary right now."

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
            
