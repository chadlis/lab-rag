import os

from .index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from pathlib import Path


def normalize(scores: list[float]) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def rrf_score(rank: int, k: int) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[dict], stopwords: set[str], cache_directory: Path = Path("cache")):
        self.documents = documents
        self.stopwords = stopwords
        self.cache_directory = Path(cache_directory)

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents, self.cache_directory)

        self.idx = InvertedIndex.load(self.cache_directory)

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        return self.idx.bm25_search(query, limit, self.stopwords)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        fetch_limit = limit * 500
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        bm25_scores = [score for _, score in bm25_results]
        sem_scores = [r["score"] for r in semantic_results]

        norm_bm25 = normalize(bm25_scores) if bm25_scores else []
        norm_sem = normalize(sem_scores) if sem_scores else []

        doc_scores: dict[int, dict] = {}
        for (doc_id, _), norm_score in zip(bm25_results, norm_bm25):
            doc_scores[doc_id] = {"bm25": norm_score, "semantic": 0.0}
        for result, norm_score in zip(semantic_results, norm_sem):
            doc_id = result["id"]
            if doc_id in doc_scores:
                doc_scores[doc_id]["semantic"] = norm_score
            else:
                doc_scores[doc_id] = {"bm25": 0.0, "semantic": norm_score}

        results = []
        for doc_id, scores in doc_scores.items():
            hybrid = alpha * scores["bm25"] + (1 - alpha) * scores["semantic"]
            doc = self.semantic_search.document_map.get(doc_id, {})
            results.append({
                "id": doc_id,
                "title": doc.get("title", ""),
                "document": doc.get("description", "")[:100],
                "bm25_score": scores["bm25"],
                "semantic_score": scores["semantic"],
                "hybrid_score": hybrid,
            })

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:limit]

    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:
        fetch_limit = limit * 500
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        doc_scores: dict[int, dict] = {}
        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            doc = self.semantic_search.document_map.get(doc_id, {})
            doc_scores[doc_id] = {
                "title": doc.get("title", ""),
                "document": doc.get("description", "")[:100],
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            if doc_id in doc_scores:
                doc_scores[doc_id]["semantic_rank"] = rank
                doc_scores[doc_id]["rrf_score"] += rrf_score(rank, k)
            else:
                doc = self.semantic_search.document_map.get(doc_id, {})
                doc_scores[doc_id] = {
                    "title": doc.get("title", ""),
                    "document": doc.get("description", "")[:100],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": rrf_score(rank, k),
                }

        results = [
            {"id": doc_id, **data} for doc_id, data in doc_scores.items()
        ]
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:limit]

