import os
# Set environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from pypdf import PdfReader
import torch
import faiss
import spacy
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
            self.sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
            self.sent_embeddings = AutoModel.from_pretrained(sent_model_name)
            
            logger.info("Loading spaCy model for sentence breaking")
            self.sentence_breaker = spacy.load("en_core_web_sm")
            
            logger.info(f"Initializing generator model: {generator_model_name}")
            self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)

            self.sent_embeddings.eval()
            
            # Set device based on availability
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            self.sent_embeddings.to(self.device)
            self.generator.to(self.device)

            # Initialize FAISS index for vector search
            self.embedding_dim = 384  # Default for MiniLM-L6
            self.db = faiss.IndexFlatL2(self.embedding_dim)
            self.mapping = None
            
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
            pdf = PdfReader(pdf_path)
            
            # Extract text from PDF
            text = ''
            for i, page in enumerate(tqdm(pdf.pages, desc="Reading PDF pages")):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                except Exception as e:
                    logger.warning(f"Error extracting text from page {i}: {str(e)}")
            
            if not text.strip():
                raise ValueError("Could not extract any text from the PDF")
                
            logger.info(f"Extracted {len(text)} characters from PDF")
            
            # Create embeddings for the text
            embeddings, self.mapping = self.get_embeddings(text)
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            
            # Clear existing index and add new embeddings
            if self.db.ntotal > 0:
                self.db.reset()
            self.db.add(embeddings.cpu().numpy())
            logger.info(f"Added {self.db.ntotal} vectors to FAISS index")
            
            return True
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise
    
    def get_embeddings(self, text):
        """Extract embeddings for text content."""
        try:
            logger.info("Breaking text into sentences")
            doc = self.sentence_breaker(text)
            sentences = [sent.text for sent in doc.sents if len(sent.text.strip()) > 0]
            logger.info(f"Found {len(sentences)} sentences")
            
            embeddings = []
            mapping = []
            
            # Process sentences in batches
            for sentence in tqdm(sentences, desc="Creating embeddings"):
                tokens = self.sent_tokenizer(sentence, return_tensors="pt", truncation=False)
                tokens = tokens['input_ids'].squeeze(0)
                
                # Split long sentences into chunks
                for i in range(0, len(tokens), 128-32):
                    chunk_tokens = tokens[i:i+128]
                    chunk_text = self.sent_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    
                    # Skip empty chunks
                    if not chunk_text.strip():
                        continue
                        
                    chunk_tokens = self.sent_tokenizer(
                        chunk_text, 
                        return_tensors="pt", 
                        truncation=False
                    ).to(self.device)
                    
                    with torch.no_grad():
                        embeded_text = self.sent_embeddings(**chunk_tokens).last_hidden_state.mean(dim=1)
                    
                    embeddings.append(embeded_text[0])
                    mapping.append(sentence)
            
            if not embeddings:
                raise ValueError("Failed to create any embeddings from the text")
                
            embeddings = torch.stack(embeddings).squeeze(1)
            return embeddings, mapping
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def get_answer(self, question, k=5):
        """Generate an answer to a question using the RAG approach."""
        try:
            if not self.mapping or self.db.ntotal == 0:
                return "Please load a document first before asking questions."
            
            logger.info(f"Generating answer for question: {question}")
            
            # Create embeddings for the question
            question_embedding, _ = self.get_embeddings(question)
            
            # Retrieve relevant passages
            indices = []
            for embedding in question_embedding:
                distances, index = self.db.search(
                    embedding.cpu().numpy().reshape(1, -1), 
                    min(k, self.db.ntotal)
                )
                indices.extend(index)
            
            # Get the retrieved sentences
            retrieved_sentences = [self.mapping[i] for i in indices[0]]
            context = " ".join(retrieved_sentences)
            
            logger.info(f"Retrieved {len(retrieved_sentences)} relevant passages")
            
            # Build prompt for the generator
            prompt = f"Summarize the following text in detail: {context}. Question: {question}"
            
            # Generate answer
            gen_tokens = self.generator_tokenizer(
                prompt, 
                return_tensors="pt",
                max_length=512, 
                truncation=True
            ).to(self.device)
            
            gen_answer = self.generator.generate(
                **gen_tokens, 
                max_length=512, 
                num_beams=4,
                early_stopping=True
            )
            
            answer = self.generator_tokenizer.decode(gen_answer[0], skip_special_tokens=True)
            logger.info("Answer generated successfully")
            
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Sorry, I encountered an error while generating an answer: {str(e)}"
            
# Initialize the RAG model if this file is run directly
# if __name__ == "__main__":
#     rag_model = RAG(
#         sent_model_name="sentence-transformers/all-MiniLM-L6-v2",
#         generator_model_name="google/flan-t5-base"
#     )