# RAG Lab
Building a RAG pipeline from scratch based on boot.dev roadmap

## Dataset
Movies with id, title, and description.
[Download the dataset](https://github.com/chadlis/lab-rag/releases/download/v0.1-data/movies.json)
Place it at `data/movies.json`.

## Setup
```bash
uv sync
cp .env.example .env  # adjust paths if needed
```

## Usage
**Build the search index** (required before searching):
```bash
uv run cli/keyword_search_cli.py build --verbose
```

**Search:**
```bash
uv run cli/keyword_search_cli.py search "brave" --limit 10 --verbose
```

## Run tests
```bash
uv run pytest tests/ -v
```


## Roadmap
### 0 - Setup
- [x] Set up the project

### 1 — Basic keyword search
- [x] Implement text preprocessing
  - [x] punctuation normalization
  - [x] tokenization
  - [x] stop word removal
- [x] Implement basic keyword search
- [x] Build an inverted index with stemming
- [x] Persist index to disk (JSON)

### 2 — TF-IDF
- [x] Implement TF-IDF retrieval
  - [x] term frequency
  - [x] inverse document frequency
  - [x] TF-IDF scoring

### 3 — Keyword Search
- [x] Implement BM25 retrieval
  - [x] term frequency saturation
  - [x] document length normalization
  - [x] BM25 scoring pipeline

### 4 — Semantic Search
- [x] Implement semantic search with embeddings
- [x] Evaluate embedding models and similarity metrics
  - [x] model selection
  - [x] vector operations
  - [x] cosine similarity
- [x] Generate document and query embeddings
- [x] Build the semantic retrieval pipeline

### 5 — Chunking
- [ ] Add document chunking for retrieval
- [ ] Support chunk overlap
- [ ] Experiment with semantic chunking
- [ ] Implement chunk-level embedding search
- [ ] Handle chunking edge cases
- [ ] Explore advanced retrieval ideas
  - [ ] ColBERT
  - [ ] late chunking

### 6 — Hybrid Search
- [ ] Compare keyword and semantic retrieval
- [ ] Implement hybrid search
- [ ] Add score normalization
- [ ] Add weighted score fusion
- [ ] Implement Reciprocal Rank Fusion (RRF)

### 7 — LLMs
- [ ] Integrate an LLM into the RAG pipeline
- [ ] Set up the Gemini API
- [ ] Improve queries with LLM assistance
  - [ ] spell correction
  - [ ] query rewriting
  - [ ] query expansion

### 8 — Reranking
- [ ] Add a reranking layer
- [ ] Experiment with LLM-based reranking
- [ ] Implement batch reranking
- [ ] Implement cross-encoder reranking
