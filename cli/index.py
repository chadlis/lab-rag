from pathlib import Path
import json

# TODO: convert CLI scripts to modules (python -m cli.keyword_search_cli) to simplify imports
try:
    from keyword_search import filter_and_stem
except ImportError:
    from cli.keyword_search import filter_and_stem

INDEX_FILENAME = "index.json"
DOCMAP_FILENAME = "docmap.json"


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}

    def build(self, movies: list[dict], stopwords: set[str]) -> None:
        for movie in movies:
            self._add_document(movie["id"], movie, stopwords)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, []))

    def save(self, cache_directory: Path) -> None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        with open(cache_directory / INDEX_FILENAME, "w") as f:
            # sets are not JSON-serializable, convert to sorted lists
            json.dump({token: sorted(doc_ids) for token, doc_ids in self.index.items()}, f)
        with open(cache_directory / DOCMAP_FILENAME, "w") as f:
            # JSON keys must be strings, int doc_ids become "1", "2", etc.
            json.dump({str(doc_id): movie for doc_id, movie in self.docmap.items()}, f)

    @classmethod
    def load(cls, cache_directory: Path) -> "InvertedIndex":
        index_filepath = Path(cache_directory) / INDEX_FILENAME
        docmap_filepath = Path(cache_directory) / DOCMAP_FILENAME

        if not index_filepath.exists():
            raise FileNotFoundError(f"{index_filepath} not found!")

        if not docmap_filepath.exists():
            raise FileNotFoundError(f"{docmap_filepath} not found!")

        instance = cls()
        with open(index_filepath) as f:
            instance.index = {token: set(doc_ids) for token, doc_ids in json.load(f).items()}
        with open(docmap_filepath) as f:
            instance.docmap = {int(doc_id): movie for doc_id, movie in json.load(f).items()}
        return instance

    def _add_document(self, doc_id: int, movie: dict, stopwords: set[str]) -> None:
        if doc_id in self.docmap:
            return
        self.docmap[doc_id] = movie
        tokens = filter_and_stem(f"{movie['title']} {movie['description']}", stopwords)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
