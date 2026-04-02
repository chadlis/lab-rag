# RAG Lab
Building a RAG pipeline from scratch based on boot.dev roadmap

## Dataset
Movies with id, title, and description.
[Download the dataset](https://github.com/chadlis/lab-rag/releases/download/v0.1-data/movies.json)

## Run
```bash
uv sync
uv run python -m lab_rag "a film about space"
```


## Roadmap
### 0 - Setup
- [x] Set up the project

### 1 — Preprocessing
- [ ] Implement text preprocessing
  - [ ] punctuation normalization
  - [ ] tokenization
  - [ ] stop word removal

### 2 — TF-IDF
- [ ] Build an inverted index
- [ ] Add boolean search
- [ ] Implement TF-IDF retrieval
  - [ ] term frequency
  - [ ] inverse document frequency
  - [ ] TF-IDF scoring

### 3 — Keyword Search
- [ ] Implement BM25 retrieval
  - [ ] term frequency saturation
  - [ ] document length normalization
  - [ ] BM25 scoring pipeline

### 4 — Semantic Search
- [ ] Implement semantic search with embeddings
- [ ] Evaluate embedding models and similarity metrics
  - [ ] model selection
  - [ ] vector operations
  - [ ] dot product similarity
  - [ ] cosine similarity
- [ ] Generate document and query embeddings
- [ ] Build the semantic retrieval pipeline
- [ ] Explore scaling strategies
  - [ ] locality-sensitive hashing
  - [ ] vector databases

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
