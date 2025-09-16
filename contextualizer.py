import logging
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Document:
    def __init__(self, id: str, title: str, content: str):
        self.id = id
        self.title = title
        self.content = content

class QwenContextualizer:
    def __init__(self, model_name="unsloth/Qwen3-14B", cache_dir="./cache"):
        logger.info(f"Loading Qwen model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        logger.info("Qwen model loaded successfully")

    def generate_context(self, document: Document, chunk: str) -> str:
        prompt = f"""
        Given the document titled '{document.title}' and the following chunk:
        '{chunk}'
        Provide a concise context (50-100 tokens) to situate this chunk within the document for improved search retrieval.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        context = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return context.strip()

class ContextualRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.chunks = []
        self.contextual_chunks = []
        self.embeddings = None
        self.bm25 = None
        self.index = None
        self.contextualizer = QwenContextualizer()

    def chunk_document(self, document: Document, chunk_size=800, overlap=100):
        tokens = word_tokenize(document.content)
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = ' '.join(chunk_tokens)
            chunks.append((chunk_text, document.id))
        return chunks

    def add_documents(self, documents: list, use_contextual=True):
        logger.info("Adding documents to retriever")
        self.documents = documents
        self.chunks = []
        self.contextual_chunks = []

        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            self.chunks.extend(doc_chunks)
            if use_contextual:
                cached_chunks = self.load_contextual_cache(doc.id)
                if cached_chunks:
                    self.contextual_chunks.extend(cached_chunks)
                else:
                    contextual_chunks = []
                    for chunk, doc_id in doc_chunks:
                        context = self.contextualizer.generate_context(doc, chunk)
                        contextual_chunks.append((f"{context} {chunk}", doc_id))
                    self.contextual_chunks.extend(contextual_chunks)
                    self.save_contextual_cache(doc.id, contextual_chunks)
            else:
                self.contextual_chunks = self.chunks

        self.index_embeddings()
        self.index_bm25()

    def index_embeddings(self):
        texts = [chunk for chunk, _ in self.contextual_chunks]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def index_bm25(self):
        tokenized_chunks = [word_tokenize(chunk.lower()) for chunk, _ in self.contextual_chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def save_contextual_cache(self, doc_id: str, contextual_chunks: list):
        cache_file = f"contextual_cache_{doc_id}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(contextual_chunks, f)

    def load_contextual_cache(self, doc_id: str):
        cache_file = f"contextual_cache_{doc_id}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def search_embedding(self, query: str, top_k=5):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [(self.contextual_chunks[i][1], distances[0][j]) for j, i in enumerate(indices[0])]

    def search_bm25(self, query: str, top_k=5):
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.contextual_chunks[i][1], scores[i]) for i in top_indices]

    def search_hybrid(self, query: str, top_k=5, alpha=0.5):
        emb_results = self.search_embedding(query, top_k=150)
        bm25_results = self.search_bm25(query, top_k=150)
        
        score_dict = {}
        for doc_id, score in emb_results:
            score_dict[doc_id] = score_dict.get(doc_id, 0) + alpha * (1 - score / np.max([s for _, s in emb_results]))
        for doc_id, score in bm25_results:
            score_dict[doc_id] = score_dict.get(doc_id, 0) + (1 - alpha) * score / np.max([s for _, s in bm25_results])
        
        sorted_results = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_results