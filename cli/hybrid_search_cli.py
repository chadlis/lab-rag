import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from lib.hybrid_search import normalize, HybridSearch
from lib.loader import load_movies, load_stopwords

load_dotenv()

_BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve(env_var: str, default: Path) -> Path:
    val = os.getenv(env_var)
    if not val:
        return default
    p = Path(val)
    return p if p.is_absolute() else _BASE_DIR / p


DATA_DIR = _resolve("DATA_DIR", _BASE_DIR / "data")
CACHE_DIR = _resolve("CACHE_DIR", _BASE_DIR / "cache")
ASSETS_DIR = _resolve("ASSETS_DIR", _BASE_DIR / "assets")

MOVIES_FILENAME = "movies.json"
STOPWORDS_FILENAME = "stopwords.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Min-max normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="scores to normalize")

    weighted_parser = subparsers.add_parser("weighted-search", help="Hybrid search with configurable alpha")
    weighted_parser.add_argument("query", type=str, help="search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="weight for semantic score (0-1)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="number of results to return")

    rrf_parser = subparsers.add_parser("rrf-search", help="Hybrid search using Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf_parser.add_argument("--limit", type=int, default=5, help="number of results to return")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if not args.scores:
                return
            for score in normalize(args.scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            movies = load_movies(DATA_DIR / MOVIES_FILENAME)
            stopwords = load_stopwords(ASSETS_DIR / STOPWORDS_FILENAME)
            hybrid = HybridSearch(movies, stopwords, CACHE_DIR)
            results = hybrid.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"  BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"  {result['document']}...")
        case "rrf-search":
            movies = load_movies(DATA_DIR / MOVIES_FILENAME)
            stopwords = load_stopwords(ASSETS_DIR / STOPWORDS_FILENAME)
            hybrid = HybridSearch(movies, stopwords, CACHE_DIR)
            results = hybrid.rrf_search(args.query, args.k, args.limit)
            for i, result in enumerate(results, start=1):
                bm25_rank = result['bm25_rank'] if result['bm25_rank'] is not None else 'N/A'
                sem_rank = result['semantic_rank'] if result['semantic_rank'] is not None else 'N/A'
                print(f"{i}. {result['title']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                print(f"  {result['document']}...")
                print()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
