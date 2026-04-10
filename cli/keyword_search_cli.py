import argparse
import os
import sys
from pathlib import Path
from collections.abc import Iterable

from dotenv import load_dotenv

from lib.keyword_search import filter_and_stem
from lib.loader import load_movies, load_stopwords
from lib.index import InvertedIndex, BM25_K1, BM25_B

load_dotenv()

_BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", _BASE_DIR / "data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", _BASE_DIR / "cache"))
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", _BASE_DIR / "assets"))

MOVIES_FILENAME = "movies.json"
STOPWORDS_FILENAME = "stopwords.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Max results to display"
    )
    search_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    build_parser = subparsers.add_parser("build", help="Build movies")
    build_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get tfidf of a term in a document")
    tfidf_parser.add_argument("document", type=int)
    tfidf_parser.add_argument("term", type=str)
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str)
    bm25_tf_parser = subparsers.add_parser("bm25tf")
    bm25_tf_parser.add_argument("document", type=int)
    bm25_tf_parser.add_argument("term", type=str)
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    bm25_tf_parser.add_argument("K1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 k1 parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Max results to display")

    args = parser.parse_args()
    match args.command:
        case "build":  # offline indexing
            print("Building the search index")
            data = load_movies(DATA_DIR / MOVIES_FILENAME)
            stopwords = load_stopwords(ASSETS_DIR / STOPWORDS_FILENAME)
            inverted_index = InvertedIndex()
            inverted_index.build(data, stopwords)
            inverted_index.save(CACHE_DIR)
            print("Index built successfully")
            if args.verbose:
                print(f"Saved in {CACHE_DIR}")
        case "search":  # online querying
            print(f"Searching for: {args.query}")
            inverted_index = load_index()
            stopwords = load_stopwords(ASSETS_DIR / STOPWORDS_FILENAME)
            query_stems = set(filter_and_stem(args.query, stopwords))
            doc_ids: set[int] = set()
            for stem in query_stems:
                doc_ids |= set(inverted_index.get_documents(stem))
            results = (inverted_index.docmap[doc_id] for doc_id in sorted(doc_ids))
            display_results(results, limit=args.limit, verbose=args.verbose)
        case "tfidf":
            print(f"Getting TF-IDF score of {args.term} in document {args.document}")
            inverted_index = load_index()
            stopwords = load_stopwords(ASSETS_DIR / STOPWORDS_FILENAME)
            tf_idf = inverted_index.get_tf_idf(args.document, args.term, stopwords)
            print(f"TF-IDF score of '{args.term}' in document '{args.document}': {tf_idf:.2f}")
        case "bm25idf":
            print(f"Getting BM25 IDF of {args.term}")
            inverted_index = load_index()
            stopwords = load_stopwords(ASSETS_DIR / STOPWORDS_FILENAME)
            bm25_idf = inverted_index.get_bm25_idf(args.term, stopwords)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25search":
            inverted_index = load_index()
            stopwords = load_stopwords(ASSETS_DIR / STOPWORDS_FILENAME)
            results = inverted_index.bm25_search(args.query, args.limit, stopwords)
            for i, (doc_id, score) in enumerate(results, 1):
                movie = inverted_index.docmap[doc_id]
                print(f"{i}. ({doc_id}) {movie['title']} - Score: {score:.2f}")
        case _:
            parser.print_help()


def load_index() -> InvertedIndex:
    try:
        return InvertedIndex.load(CACHE_DIR)
    except FileNotFoundError as e:
        print(e)
        print("Index not found. Run 'build' first!")
        sys.exit(1)


def display_results(results: Iterable[dict], limit: int = 5, verbose=False) -> None:
    found = False
    for i, result in enumerate(results):
        if i >= limit:
            break
        found = True
        print(f"{i + 1}. {result['title']}")
        if verbose: 
            print(f"Description: {result['description']}\n\n")
    if not found:
        print("No results found.")


if __name__ == "__main__":
    main()
