from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from pypdf import PdfReader
import torch
import faiss
import spacy

class RAG():
    def __init__(self, sent_model_name, generator_model_name):
        self.sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
        self.sent_embeddings = AutoModel.from_pretrained(sent_model_name)
        self.sentence_breaker = spacy.load("en_core_web_sm")
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)

        self.sent_embeddings.eval()

        self.sent_embeddings.to("cuda")
        self.generator.to("cuda")

        self.db = faiss.IndexFlatL2(384)
        self.mapping = None


    def load_pdf(self, pdf_path):
        pdf = PdfReader(pdf_path)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        
        embeddings, self.mapping = self.get_embeddings(text)
        print(embeddings.shape)
        self.db.add(embeddings.cpu().numpy())

    
    def get_embeddings(self, text):
        sentences = [sent.text for sent in self.sentence_breaker(text).sents if len(sent.text) > 0]
        embeddings = []
        mapping = []
        for sentence in sentences:
            tokens = self.sent_tokenizer(sentence, return_tensors="pt", truncation=False)
            tokens = tokens['input_ids'].squeeze(0)
            for i in range(0, len(tokens), 128-32):
                chunk_tokens = tokens[i:i+128]
                chunk_text = self.sent_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunk_tokens = self.sent_tokenizer(chunk_text, return_tensors="pt", truncation=False).to('cuda')
                with torch.no_grad():
                    embeded_text = self.sent_embeddings(**chunk_tokens).last_hidden_state.mean(dim=1)
                embeddings.append(embeded_text[0])
                mapping.append(sentence)
        embeddings = torch.stack(embeddings).squeeze(1)
        return embeddings, mapping
    
    def get_answer(self, question, k=5):
        question_embedding,_ = self.get_embeddings(question)
        indices = []
        for embedding in question_embedding:
            _,index = self.db.search(embedding.cpu().numpy().reshape(1,-1), k)
            indices.extend(index)
        retrieved_sentences = [self.mapping[i] for i in indices[0]]
        retrieved_sentences = " ".join(retrieved_sentences)
        print(f"Retrieved sentences: {retrieved_sentences}")
        context = f"Summarize the following text in detail: {retrieved_sentences}. Question: {question}"

        gen_tokens = self.generator_tokenizer(context, return_tensors="pt").to("cuda")
        gen_answer = self.generator.generate(**gen_tokens, max_length=512, num_beams=4,early_stopping=True)
        gen_answer = self.generator_tokenizer.decode(gen_answer[0], skip_special_tokens=True)
        return gen_answer