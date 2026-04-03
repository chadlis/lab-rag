import argparse
import os
from pathlib import Path
from collections.abc import Iterable

from dotenv import load_dotenv

from keyword_search import filter_and_stem
from loader import load_movies, load_stopwords
from index import InvertedIndex

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache"))
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "assets"))

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

    args = parser.parse_args()
    match args.command:
        case "build":  # offline indexing
            print("Building the search index")
            data = load_movies(DATA_DIR / MOVIES_FILENAME)
            stopwords = load_stopwords(ASSETS_DIR / STOPWORDS_FILENAME)
            inverted_index = InvertedIndex()
            inverted_index.build(data, stopwords)
            inverted_index.save(CACHE_DIR, args.verbose)
        case "search":  # online querying
            print(f"Searching for: {args.query}")
            try:
                inverted_index = InvertedIndex.load(CACHE_DIR)
            except FileNotFoundError as e:
                print(e)
                print("Index not found. Run 'build' first!")
                return
            stopwords = load_stopwords(ASSETS_DIR / STOPWORDS_FILENAME)
            query_stems = filter_and_stem(args.query, stopwords)
            doc_ids: set[int] = set()
            for stem in query_stems:
                doc_ids |= set(inverted_index.get_documents(stem))
            results = (inverted_index.docmap[doc_id] for doc_id in sorted(doc_ids))
            display_results(results, limit=args.limit, verbose=args.verbose)
        case _:
            parser.print_help()


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
