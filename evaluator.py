import logging
import time
import json
import numpy as np
from nltk.tokenize import sent_tokenize
from .contextualizer import Document, ContextualRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationResult:
    def __init__(self, method: str, top_k: int, recall_at_k: float, failure_rate: float, avg_retrieval_time: float, total_queries: int):
        self.method = method
        self.top_k = top_k
        self.recall_at_k = recall_at_k
        self.failure_rate = failure_rate
        self.avg_retrieval_time = avg_retrieval_time
        self.total_queries = total_queries

class EvaluationFramework:
    def create_synthetic_queries(self, documents: list, num_queries: int = 50) -> list:
        """Create synthetic queries for evaluation"""
        logger.info(f"Creating {num_queries} synthetic queries...")
        queries = []
        for i, doc in enumerate(documents[:num_queries]):
            sentences = sent_tokenize(doc.content)
            if len(sentences) >= 2:
                if "revenue" in doc.content.lower():
                    query_text = f"What was the revenue growth for {doc.title.split()[0]} in 2023?"
                elif "error code" in doc.content.lower():
                    query_text = f"What causes error code TS-999 in {doc.title}?"
                else:
                    query_text = f"What are the key findings in {doc.title}?"
                query = {
                    'id': f'query_{i}',
                    'text': query_text,
                    'relevant_doc_ids': [doc.id],
                    'doc_title': doc.title
                }
                queries.append(query)
        # Add an irrelevant query to test failure cases
        queries.append({
            'id': 'query_irrelevant_1',
            'text': 'What is the capital of France?',
            'relevant_doc_ids': [],
            'doc_title': 'None'
        })
        logger.info(f"Created {len(queries)} synthetic queries")
        return queries

    def evaluate_retrieval(self, retriever: ContextualRetriever, queries: list, method: str, top_k: int = 5):
        """Evaluate retrieval performance for a given method"""
        logger.info(f"Evaluating retrieval method: {method}")
        total_queries = len(queries)
        if total_queries == 0:
            return EvaluationResult(method, top_k, 0.0, 1.0, 0.0, 0)
        
        recall_sum = 0
        failure_count = 0
        retrieval_times = []
        
        for query in queries:
            start_time = time.time()
            if method == "embedding":
                results = retriever.search_embedding(query['text'], top_k=top_k)
            elif method == "bm25":
                results = retriever.search_bm25(query['text'], top_k=top_k)
            else:  # hybrid
                results = retriever.search_hybrid(query['text'], top_k=top_k)
            
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)
            
            retrieved_doc_ids = set(doc_id for doc_id, _ in results)
            relevant_doc_ids = set(query['relevant_doc_ids'])
            
            logger.info(f"Query: {query['text']}, Retrieved Docs: {retrieved_doc_ids}, Expected: {relevant_doc_ids}")
            
            if relevant_doc_ids:
                if retrieved_doc_ids & relevant_doc_ids:
                    recall_sum += 1
                else:
                    failure_count += 1
                    logger.warning(f"Query {query['id']} failed to retrieve relevant docs: {query['text']}")
        
        recall_at_k = recall_sum / total_queries if total_queries > 0 else 0.0
        failure_rate = failure_count / total_queries if total_queries > 0 else 1.0
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0
        
        return EvaluationResult(method, top_k, recall_at_k, failure_rate, avg_retrieval_time, total_queries)