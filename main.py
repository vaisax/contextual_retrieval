import logging
import json
import time
import os
from .contextualizer import Document, ContextualRetriever
from .evaluator import EvaluationFramework, EvaluationResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('logs/contextual_retrieval.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def create_sample_documents():
    """Create sample documents for the demo"""
    sample_docs = [
        Document("doc_1", "ACME Corp Q2 2023 Financial Report", "ACME Corp's revenue grew by 3% to $320M in Q2 2023 compared to $310M in Q1 2023..."),
        Document("doc_2", "Technical Documentation for Error Code TS-999", "Error code TS-999 occurs due to a timeout in the API server..."),
        Document("doc_3", "Research Paper on Climate Change", "Climate change is impacting marine ecosystems through rising sea temperatures..."),
        Document("doc_4", "Software Best Practices Guide", "Adopting agile methodologies improves software development efficiency..."),
        Document("doc_5", "Marketing Strategy Report 2023", "The 2023 marketing strategy focuses on digital channels..."),
    ]
    return sample_docs

def run_demo():
    """Run a simple demo showing context generation"""
    logger.info("Running Contextual Retrieval Demo")
    documents = create_sample_documents()
    retriever = ContextualRetriever()
    
    # Add a single document with context
    retriever.add_documents([documents[0]], use_contextual=True)
    
    # Show a sample chunk and its context
    chunk, doc_id = retriever.chunks[0]
    contextual_chunk, _ = retriever.contextual_chunks[0]
    print("\nSample Chunk:")
    print(chunk[:200] + "...")
    print("\nWith Context:")
    print(contextual_chunk[:300] + "...")
    
    # Demo a search
    query = "What was ACME Corp's revenue growth in 2023?"
    results = retriever.search_hybrid(query, top_k=3)
    print("\nSample Search Results:")
    for doc_id, score in results:
        doc = next(d for d in documents if d.id == doc_id)
        print(f"Document: {doc.title}, Score: {score:.4f}")

def run_comparison_experiment():
    """Run the main comparison experiment"""
    logger.info("Starting Contextual Retrieval Comparison Experiment")
    
    documents = create_sample_documents()
    logger.info(f"Created {len(documents)} sample documents")
    
    evaluator = EvaluationFramework()
    queries = evaluator.create_synthetic_queries(documents, num_queries=50)
    
    results = []
    
    logger.info("\n" + "="*60)
    logger.info("Testing Traditional Retrieval")
    logger.info("="*60)
    
    traditional_retriever = ContextualRetriever()
    traditional_retriever.add_documents(documents, use_contextual=False)
    
    for method in ["embedding", "bm25", "hybrid"]:
        result = evaluator.evaluate_retrieval(traditional_retriever, queries, method, top_k=5)
        results.append(result)
    
    logger.info("\n" + "="*60)
    logger.info("Testing Contextual Retrieval")
    logger.info("="*60)
    
    contextual_retriever = ContextualRetriever()
    contextual_retriever.add_documents(documents, use_contextual=True)
    
    for method in ["embedding", "bm25", "hybrid"]:
        method_name = f"contextual_{method}"
        result = evaluator.evaluate_retrieval(contextual_retriever, queries, method, top_k=5)
        result.method = method_name
        results.append(result)
    
    print("\n" + "="*80)
    print("CONTEXTUAL RETRIEVAL PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"{'Method':<20} {'Recall@5':<12} {'Failure Rate':<15} {'Avg Time (ms)':<15} {'Improvement':<12}")
    print("-" * 80)
    
    baseline_results = {r.method: r for r in results if not r.method.startswith('contextual_')}
    contextual_results = {r.method.replace('contextual_', ''): r for r in results if r.method.startswith('contextual_')}
    
    for method in ["embedding", "bm25", "hybrid"]:
        if method in baseline_results and method in contextual_results:
            baseline = baseline_results[method]
            contextual = contextual_results[method]
            
            if baseline.failure_rate == 0:
                if contextual.failure_rate == 0:
                    failure_improvement = 0.0
                    improvement_str = "Perfect (0%)"
                else:
                    failure_improvement = float('inf')
                    improvement_str = "Worse (∞%)"
            else:
                failure_improvement = ((baseline.failure_rate - contextual.failure_rate) / baseline.failure_rate) * 100
                improvement_str = f"+{failure_improvement:.1f}%"
            
            logger.info(f"Method: {method}, Baseline Failure Rate: {baseline.failure_rate:.3f}, Contextual Failure Rate: {contextual.failure_rate:.3f}")
            
            print(f"{baseline.method:<20} {baseline.recall_at_k:<12.3f} {baseline.failure_rate:<15.3f} {baseline.avg_retrieval_time*1000:<15.1f} {'Baseline':<12}")
            print(f"{'contextual_'+method:<20} {contextual.recall_at_k:<12.3f} {contextual.failure_rate:<15.3f} {contextual.avg_retrieval_time*1000:<15.1f} {improvement_str:<12}")
            print("-" * 80)
    
    print("\nSUMMARY:")
    print("-" * 40)
    
    hybrid_baseline = baseline_results["hybrid"].failure_rate
    hybrid_contextual = contextual_results["hybrid"].failure_rate
    
    if hybrid_baseline == 0:
        if hybrid_contextual == 0:
            overall_improvement = 0.0
            improvement_str = "No improvement (both perfect)"
        else:
            overall_improvement = float('inf')
            improvement_str = "Worse (contextual has failures)"
    else:
        overall_improvement = ((hybrid_baseline - hybrid_contextual) / hybrid_baseline) * 100
        improvement_str = f"{overall_improvement:.1f}% reduction in failure rate"
    
    print(f"Best Traditional Method (Hybrid): {hybrid_baseline:.3f} failure rate")
    print(f"Best Contextual Method (Hybrid): {hybrid_contextual:.3f} failure rate")
    print(f"Overall Improvement: {improvement_str}")
    
    results_data = {
        'experiment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'documents_count': len(documents),
        'queries_count': len(queries),
        'results': [
            {
                'method': r.method,
                'top_k': r.top_k,
                'recall_at_k': r.recall_at_k,
                'failure_rate': r.failure_rate,
                'avg_retrieval_time': r.avg_retrieval_time,
                'total_queries': r.total_queries
            } for r in results
        ]
    }
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info("Experiment completed. Results saved to experiment_results.json")
    return results

def main():
    print("Welcome to Contextual Retrieval Demo!")
    choice = input("Run full experiment (requires GPU, ~10-15 min)? [y/n]: ").lower()
    
    if choice == 'y':
        try:
            results = run_comparison_experiment()
        except Exception as e:
            logger.error(f"Error during experiment execution: {str(e)}")
            print(f"Error: {str(e)}")
    else:
        run_demo()

if __name__ == "__main__":
    main()