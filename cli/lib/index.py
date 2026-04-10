from pathlib import Path
from statistics import mean
import json
from collections import Counter
import math
from tqdm import tqdm
from .keyword_search import filter_and_stem

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    _INDEX_FILENAME = "index.json"
    _DOCMAP_FILENAME = "docmap.json"
    _TERM_FREQUENCIES_FILENAME = "term_frequencies.json"
    _INVERSE_DOC_FREQUENCIES_FILENAME = "inverse_doc_frequencies.json"
    _BM25_IDF_FILENAME = "bm25_idf.json"
    _BM25_TF_FILENAME = "bm25_tf.json"
    _DOC_LENGTHS_FILENAME = "doc_lengths.json"

    def __init__(self):
        self.reset()
        
        
    def reset(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.bm25_tf: dict[int, dict[str, float]] = {}
        self.inverse_doc_frequencies: dict[str, float] = {}
        self.bm25_idf: dict[str, float] = {}
        self.doc_lengths: dict[int, int] = {}
        self.__avg_doc_length: float = 0
    
    def build(self, movies: list[dict], stopwords: set[str]) -> None:
        for movie in tqdm(movies):
            self._add_document(movie["id"], movie, stopwords)
        self.__avg_doc_length = mean(self.doc_lengths.values())
        for doc_id, tf_counts in self.term_frequencies.items():
            length_norm = 1 - BM25_B + BM25_B * (self.doc_lengths[doc_id] / self.__avg_doc_length)
            self.bm25_tf[doc_id] = {term: ((tf*(BM25_K1+1)) / (tf + (BM25_K1* length_norm))) for term, tf in tf_counts.items()}

        total_doc_count = len(self.docmap)
        for token, doc_ids in self.index.items():
            self.inverse_doc_frequencies[token] = math.log((total_doc_count + 1) / (len(doc_ids) + 1))
            self.bm25_idf[token] = math.log((total_doc_count - len(doc_ids) + 0.5)/(len(doc_ids) + 0.5)  + 1)
        

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
    
    def get_bm25_idf(self, term: str, stopwords) -> float:
        token = filter_and_stem(term, stopwords)
        if len(token) != 1:
            raise Exception("Term results in more than one token")
        return self.bm25_idf.get(token[0], 0)

    def get_bm25(self, doc_id: int, term: str, stopwords) -> float:
        token = filter_and_stem(term, stopwords)
        if len(token) != 1:
            raise Exception("Term results in more than one token")
        bm25_idf = self.bm25_idf.get(token[0], 0)
        if doc_id not in self.bm25_tf:
            raise KeyError(f"Document {doc_id} not found in index")
        bm25tf = self.bm25_tf[doc_id].get(token[0],0)
        return bm25tf * bm25_idf
    
    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.bm25_tf.get(doc_id, {}).get(term, 0)
        bm25_idf = self.bm25_idf.get(term, 0)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int, stopwords: set) -> list[tuple[int, float]]:
        tokens = filter_and_stem(query, stopwords)
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            total = 0.0
            for token in tokens:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:limit]

    
        

    def save(self, cache_directory: Path) -> None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        with open(cache_directory / self._INDEX_FILENAME, "w") as f:
            json.dump({token: sorted(doc_ids) for token, doc_ids in self.index.items()}, f)
        with open(cache_directory / self._DOCMAP_FILENAME, "w") as f:
            json.dump({str(doc_id): movie for doc_id, movie in self.docmap.items()}, f)
        with open(cache_directory / self._TERM_FREQUENCIES_FILENAME, "w") as f:
            json.dump({str(doc_id): counts for doc_id, counts in self.term_frequencies.items()}, f)
        with open(cache_directory / self._INVERSE_DOC_FREQUENCIES_FILENAME, "w") as f:
            json.dump(self.inverse_doc_frequencies, f)
        with open(cache_directory / self._BM25_IDF_FILENAME, "w") as f:
            json.dump(self.bm25_idf, f)
        with open(cache_directory / self._BM25_TF_FILENAME, "w") as f:
            json.dump(self.bm25_tf, f)
        with open(cache_directory / self._DOC_LENGTHS_FILENAME, "w") as f:
            json.dump(self.doc_lengths, f)

    @classmethod
    def load(cls, cache_directory: Path) -> "InvertedIndex":
        cache_directory = Path(cache_directory)
        index_filepath = cache_directory / cls._INDEX_FILENAME
        docmap_filepath = cache_directory / cls._DOCMAP_FILENAME
        term_frequencies_filepath = cache_directory / cls._TERM_FREQUENCIES_FILENAME
        inverse_doc_frequencies_filepath = cache_directory / cls._INVERSE_DOC_FREQUENCIES_FILENAME
        bm25_idf_filepath = cache_directory / cls._BM25_IDF_FILENAME
        bm25_tf_filepath = cache_directory / cls._BM25_TF_FILENAME
        doc_lengths_filepath = cache_directory / cls._DOC_LENGTHS_FILENAME
        
        if not index_filepath.exists():
            raise FileNotFoundError(f"{index_filepath} not found!")
        if not docmap_filepath.exists():
            raise FileNotFoundError(f"{docmap_filepath} not found!")
        
        if not term_frequencies_filepath.exists():
            raise FileNotFoundError(f"{term_frequencies_filepath} not found!")
        if not inverse_doc_frequencies_filepath.exists():
            raise FileNotFoundError(f"{inverse_doc_frequencies_filepath} not found!")

        if not bm25_tf_filepath.exists():
            raise FileNotFoundError(f"{bm25_tf_filepath} not found!")
        if not bm25_idf_filepath.exists():
            raise FileNotFoundError(f"{bm25_idf_filepath} not found!")
        if not doc_lengths_filepath.exists():
            raise FileNotFoundError(f"{doc_lengths_filepath} not found!")

        instance = cls()
        with open(index_filepath) as f:
            instance.index = {token: set(doc_ids) for token, doc_ids in json.load(f).items()}
        with open(docmap_filepath) as f:
            instance.docmap = {int(doc_id): movie for doc_id, movie in json.load(f).items()}
        with open(term_frequencies_filepath) as f:
            instance.term_frequencies = {int(doc_id): Counter(counts) for doc_id, counts in json.load(f).items()}
        with open(inverse_doc_frequencies_filepath) as f:
            instance.inverse_doc_frequencies = {term: float(frequency) for term, frequency in json.load(f).items()}
        with open(bm25_idf_filepath) as f:
            instance.bm25_idf = {term: float(frequency) for term, frequency in json.load(f).items()}
        with open(bm25_tf_filepath) as f:
            instance.bm25_tf = {int(doc_id): dict(counts) for doc_id, counts in json.load(f).items()}
        with open(doc_lengths_filepath) as f:
            instance.doc_lengths = {int(doc_id): int(length) for doc_id, length in json.load(f).items()}
        return instance

    def _add_document(self, doc_id: int, movie: dict, stopwords: set[str]) -> None:
        if doc_id in self.docmap:
            return
        
        self.docmap[doc_id] = movie
        tokens = filter_and_stem(f"{movie['title']} {movie['description']}", stopwords)
        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id] = Counter(tokens)

        for token in set(tokens):
            self.index.setdefault(token, set()).add(doc_id)
