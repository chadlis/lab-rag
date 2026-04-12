import httpx
from huggingface_hub.utils import set_client_factory
from torch import Tensor #!TODO: delete this workaround if no more needed
set_client_factory(lambda: httpx.Client(follow_redirects=True, timeout=None, verify=False))
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
import re

SCORE_PRECISION = 4

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    
def chunk_text(text: str, chunk_size: int=200, overlap: int=40)-> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size-overlap):
        chunks.append(' '.join(words[i:i + chunk_size]))
        if (i + chunk_size) >= len(words):
            break
    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap : int = 0) -> list[str]:
    regex = r"(?<=[.!?])\s+"
    sentences = re.split(regex, text)
    chunks = []
    for i in range(0, len(sentences), max_chunk_size - overlap):
        chunks.append(" ".join(sentences[i:i+max_chunk_size]))
        if (i + max_chunk_size) >= len(sentences):
            break
    return chunks
    
    
class SemanticSearch:
    _EMBEDDINGS_FILENAME = "movie_embeddings.npy"
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.reset()
    
    def reset(self):
        self.embeddings: np.ndarray | Tensor | None = None
        self.documents: list[dict] = []
        
    def build(self, documents: list[dict]) -> None:
        self.documents = documents
        descriptions: list[str] = []
        for doc in self.documents:
            descriptions.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(descriptions, show_progress_bar=True)
        
    def save(self, cache_directory: Path):
        cache_directory = Path(cache_directory)
        cache_directory.mkdir(exist_ok=True, parents=True)
        if self.embeddings is None:
            raise ValueError("Embeddings are empty or None!")
        np.save(cache_directory/self._EMBEDDINGS_FILENAME, self.embeddings)
    
    @classmethod
    def load_or_create_embeddings(cls, documents: list[dict], cache_directory: Path):
        cache_directory = Path(cache_directory)
        instance = cls()
        instance.documents = documents
        embeddings_filepath = cache_directory / cls._EMBEDDINGS_FILENAME
        if embeddings_filepath.exists():
            instance.embeddings = np.load(embeddings_filepath)
            if instance.embeddings is None:
                raise RuntimeError("Embeddings failed to load.")
            if len(instance.embeddings) != len(documents):
                raise RuntimeError(
                    f"Cached embeddings {len(instance.embeddings)} don't match documents {len(documents)}"
                )
        else:
            instance.build(documents)
            instance.save(cache_directory)
        return instance
        
    def generate_embedding(self, text: str):
        text = text.strip()
        if text == "":
            raise ValueError("Input is empty or containing only white spaces")
        return self.model.encode([text])[0]
    
    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarities = self.embeddings @ query_embedding / (norms * query_norm + 1e-10)
        top_indices = np.argsort(similarities)[::-1][:limit]
        return [
            {"score": float(similarities[i]), "title": self.documents[i]["title"], "description": self.documents[i]["description"]}
            for i in top_indices
        ]
        
class ChunkedSemanticSearch(SemanticSearch):
    _CHUNK_EMBEDDING_FILENAME = "chunk_embedding.npy"
    _CHUNK_METADATA_FILENAME = "chunk_metadata.json"
    def __init__(self, model_name : str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: None | np.ndarray | Tensor = None
        self.chunk_metadata: list[dict] = []
        self.document_map = {}
    
    def build_chunk_embeddings(self, documents: list[dict], cache_directory: Path) -> np.ndarray|Tensor:
        cache_directory = Path(cache_directory)
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []
        for movie_idx, movie in enumerate(self.documents):
            if not movie["description"].strip():
                continue
            doc_chunks = semantic_chunk_text(movie["description"], max_chunk_size=4, overlap=1)
            total_chunks = len(doc_chunks)
            all_chunks.extend(doc_chunks)
            chunk_metadata.extend(
                {
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                }
                for chunk_idx in range(total_chunks)
            )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        cache_directory.mkdir(parents=True, exist_ok=True)
        np.save(cache_directory / self._CHUNK_EMBEDDING_FILENAME, self.chunk_embeddings)
        with open(cache_directory / self._CHUNK_METADATA_FILENAME, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict], cache_directory: Path) -> np.ndarray|Tensor:
        cache_directory = Path(cache_directory)
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        embeddings_path = cache_directory / self._CHUNK_EMBEDDING_FILENAME
        metadata_path = cache_directory / self._CHUNK_METADATA_FILENAME
        if embeddings_path.exists() and metadata_path.exists():
            self.chunk_embeddings = np.load(embeddings_path)
            with open(metadata_path) as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents, cache_directory)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or not self.chunk_metadata:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict] = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append({
                "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score,
            })
        movie_scores: dict[int, dict] = {}
        for cs in chunk_scores:
            movie_idx = cs["movie_idx"]
            if movie_idx not in movie_scores or cs["score"] > movie_scores[movie_idx]["score"]:
                movie_scores[movie_idx] = cs
        sorted_movies = sorted(movie_scores.values(), key=lambda x: x["score"], reverse=True)[:limit]
        results = []
        for entry in sorted_movies:
            doc = self.documents[entry["movie_idx"]]
            metadata = self.chunk_metadata[entry["chunk_idx"]] if entry["chunk_idx"] < len(self.chunk_metadata) else {}
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(entry["score"], SCORE_PRECISION),
                "metadata": metadata or {},
            })
        return results
