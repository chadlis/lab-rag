
import httpx
from huggingface_hub.utils import set_client_factory #!TODO: delete this workaround if no more needed
set_client_factory(lambda: httpx.Client(follow_redirects=True, timeout=None, verify=False))
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

class SemanticSearch:
    _EMBEDDINGS_FILENAME = "movie_embeddings.npy"
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.reset()
    
    def reset(self):
        self.embeddings: np.ndarray | None = None
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
        
