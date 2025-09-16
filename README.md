# Contextual Retrieval with Qwen3-14B-FP8 Demo

This project demonstrates **Contextual Retrieval**, a technique to enhance Retrieval-Augmented Generation (RAG) systems by adding chunk-specific context to document chunks before indexing, as described in Anthropic's [Contextual Retrieval article](https://www.anthropic.com/news/contextual-retrieval). The implementation uses the `unsloth/Qwen3-14B-FP8` model for context generation, `sentence-transformers` for embeddings, and `rank-bm25` for term-based search.

## Features
- **Document Chunking**: Splits documents into overlapping chunks (800 tokens by default).
- **Context Generation**: Uses Qwen3-14B to generate concise context (50-100 tokens) for each chunk.
- **Hybrid Search**: Combines semantic embeddings (`all-MiniLM-L6-v2`) and BM25 for robust retrieval.
- **Evaluation**: Compares traditional vs. contextual retrieval with metrics like Recall@5 and failure rate.
- **Sample Documents**: Includes 5 diverse documents (financial reports, tech docs, etc.) for testing.

## Installation

### Prerequisites
- Python 3.8+
- GPU with 8GB+ VRAM (recommended for Qwen3-14B)
- Git

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/vaisax/contextual_retrieval.git
   cd contextual_retrieval
   ```
2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
3. Activate the virtual environment:
   ```bash
   source contextual_retrieval_env/bin/activate
   ```

### Dependencies
Key dependencies (full list in `requirements.txt`):
- `torch`
- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `rank-bm25`
- `nltk`

## Usage

Run the main script:
```bash
python -m contextual_retrieval.main
```

### Options
- **Demo Mode**: Shows context generation and a sample search (lightweight, no GPU required if cached).
- **Full Experiment**: Compares traditional vs. contextual retrieval (GPU recommended, ~10-15 min).
  - Generates 50 synthetic queries
  - Evaluates embedding, BM25, and hybrid methods
  - Saves results to `experiment_results.json`

## Project Structure
```
contextual_retrieval/
├── contextual_retrieval/
│   ├── __init__.py
│   ├── contextualizer.py  # Document chunking, context generation, indexing
│   ├── evaluator.py       # Query generation and evaluation
│   ├── main.py            # Main script for demo/experiment
├── data/
│   ├── sample_documents/  # Sample documents
├── logs/                  # Log files
├── setup.sh               # Setup script
├── requirements.txt        # Dependencies
├── README.md              # This file
```

## Results
The experiment outputs:
- **Recall@5**: Percentage of queries with at least one relevant document in top 5 results.
- **Failure Rate**: Percentage of queries failing to retrieve relevant documents.
- **Retrieval Time**: Average time per query.
- **Qualitative Examples**: Shows how context improves retrieval.

Sample output:
```
================================================================================
CONTEXTUAL RETRIEVAL PERFORMANCE COMPARISON
================================================================================
Method               Recall@5     Failure Rate    Avg Time (ms)   Improvement 
--------------------------------------------------------------------------------
embedding            0.820        0.180           4.8             Baseline    
contextual_embedding 0.900        0.100           5.1             +44.4%      
--------------------------------------------------------------------------------
bm25                 0.780        0.220           0.2             Baseline    
contextual_bm25      0.860        0.140           0.2             +36.4%      
--------------------------------------------------------------------------------
hybrid               0.880        0.120           2.6             Baseline    
contextual_hybrid    0.940        0.060           2.8             +50.0%      
--------------------------------------------------------------------------------
```

## Notes
- **Resource Intensive**: Qwen3-14B requires significant compute. Use a smaller model (e.g., Qwen3-7B) for CPU-only setups.
- **Caching**: Contextual chunks are cached to `contextual_cache_<doc_id>.pkl`.
- **Customization**: Adjust chunk size, overlap, or prompts in `contextualizer.py`.

## Troubleshooting
- **Out of Memory**: Reduce chunk size or use a smaller model.
- **Dependency Issues**: Ensure `setup.sh` ran successfully or install manually with `pip install -r requirements.txt`.
- **Zero Failure Rates**: If all methods show 0% failure, add more documents or use more complex queries (see `evaluator.py`).

## Contributing
Fork the repo, make changes, and submit a pull request. Issues and suggestions are welcome!

## License
Go kray!