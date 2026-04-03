from pathlib import Path
import json
from collections import Counter
import math
from tqdm import tqdm 
# TODO: convert CLI scripts to modules (python -m cli.keyword_search_cli) to simplify imports
try:
    from keyword_search import filter_and_stem
except ImportError:
    from cli.keyword_search import filter_and_stem

INDEX_FILENAME = "index.json"
DOCMAP_FILENAME = "docmap.json"
TERM_FREQUENCIES_FILENAME = "term_frequencies.json"
INVERSE_DOC_FREQUENCIES_FILENAME = "inverse_doc_frequencies.json"

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.inverse_doc_frequencies: dict[str, float] = {}

    def build(self, movies: list[dict], stopwords: set[str]) -> None:
        for movie in tqdm(movies):
            self._add_document(movie["id"], movie, stopwords)
        total_doc_count = len(self.docmap)
        for token, doc_ids in self.index.items():
            self.inverse_doc_frequencies[token] = math.log((total_doc_count + 1) / (len(doc_ids) + 1))
        

    def get_documents(self, stem: str) -> list[int]:
        return sorted(self.index.get(stem, []))
    

    def get_tf_idf(self, doc_id: int, term: str, stopwords) -> float:
        token = filter_and_stem(term, stopwords)
        if len(token) != 1:
            raise Exception("Term results in more than one token")
        if doc_id not in self.term_frequencies:
            raise KeyError(f"Document {doc_id} not found in index")
        tf = self.term_frequencies[doc_id][token[0]]
        idf = self.inverse_doc_frequencies.get(token[0], 0)
        return (tf * idf)
        
    def save(self, cache_directory: Path) -> None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        with open(cache_directory / INDEX_FILENAME, "w") as f:
            json.dump({token: sorted(doc_ids) for token, doc_ids in self.index.items()}, f)
        with open(cache_directory / DOCMAP_FILENAME, "w") as f:
            json.dump({str(doc_id): movie for doc_id, movie in self.docmap.items()}, f)
        with open(cache_directory / TERM_FREQUENCIES_FILENAME, "w") as f:
            json.dump({str(doc_id): counts for doc_id, counts in self.term_frequencies.items()}, f)
        with open(cache_directory / INVERSE_DOC_FREQUENCIES_FILENAME, "w") as f:
            json.dump(self.inverse_doc_frequencies, f)

    @classmethod
    def load(cls, cache_directory: Path) -> "InvertedIndex":
        index_filepath = Path(cache_directory) / INDEX_FILENAME
        docmap_filepath = Path(cache_directory) / DOCMAP_FILENAME
        term_frequencies_filepath = Path(cache_directory) / TERM_FREQUENCIES_FILENAME
        inverse_doc_frequencies_filepath = Path(cache_directory) / INVERSE_DOC_FREQUENCIES_FILENAME
        
        if not index_filepath.exists():
            raise FileNotFoundError(f"{index_filepath} not found!")

        if not docmap_filepath.exists():
            raise FileNotFoundError(f"{docmap_filepath} not found!")
        
        if not term_frequencies_filepath.exists():
            raise FileNotFoundError(f"{term_frequencies_filepath} not found!")

        if not inverse_doc_frequencies_filepath.exists():
            raise FileNotFoundError(f"{inverse_doc_frequencies_filepath} not found!")

        instance = cls()
        with open(index_filepath) as f:
            instance.index = {token: set(doc_ids) for token, doc_ids in json.load(f).items()}
        with open(docmap_filepath) as f:
            instance.docmap = {int(doc_id): movie for doc_id, movie in json.load(f).items()}
        with open(term_frequencies_filepath) as f:
            instance.term_frequencies = {int(doc_id): Counter(counts) for doc_id, counts in json.load(f).items()}
        with open(inverse_doc_frequencies_filepath) as f:
            instance.inverse_doc_frequencies = {term: float(frequency) for term, frequency in json.load(f).items()}
        return instance

    def _add_document(self, doc_id: int, movie: dict, stopwords: set[str]) -> None:
        if doc_id in self.docmap:
            return
        self.docmap[doc_id] = movie
        tokens = filter_and_stem(f"{movie['title']} {movie['description']}", stopwords)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in set(tokens):
            self.index.setdefault(token, set()).add(doc_id)

