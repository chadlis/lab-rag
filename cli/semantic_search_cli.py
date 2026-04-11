#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from lib.loader import load_movies
from lib.semantic_search import embed_text, SemanticSearch

load_dotenv()

_BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", _BASE_DIR / "data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", _BASE_DIR / "cache"))
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", _BASE_DIR / "assets"))

MOVIES_FILENAME = "movies.json"
def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")
    build_parser = subparsers.add_parser("build", help="build the embeddings")
    build_parser.add_argument("--verbose", action="store_true")
    subparsers.add_parser("verify_embeddings", help="Verify the semantic search model")
    embed_parser = subparsers.add_parser("embed_text", help="Embed the given text")
    embed_parser.add_argument("text", type=str, help="the text to embed")
    embedquery_parser = subparsers.add_parser("embedquery")
    embedquery_parser.add_argument("query")
    search_parser = subparsers.add_parser("search", help="Search movies by semantic similarity")
    search_parser.add_argument("query", type=str, help="the search query")
    search_parser.add_argument("--limit", type=int, default=5, help="number of results to return")
    args = parser.parse_args()

    match args.command:
        case "build":  # offline indexing
            print("Building the search index")
            data = load_movies(DATA_DIR / MOVIES_FILENAME)
            semantic_search = SemanticSearch()
            semantic_search.build(data)
            semantic_search.save(CACHE_DIR)
            print("Embeddings built successfully")
            if args.verbose:
                print(f"Saved in {CACHE_DIR}")
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            documents = load_movies(DATA_DIR / MOVIES_FILENAME)
            semantic_search = SemanticSearch.load_or_create_embeddings(documents, CACHE_DIR)
            print(f"Number of docs:   {len(documents)}")
            print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")
        case "embedquery":
            embed_text(args.query)
        case "search":
            movies = load_movies(DATA_DIR / MOVIES_FILENAME)
            semantic_search = SemanticSearch.load_or_create_embeddings(movies, CACHE_DIR)
            results = semantic_search.search(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {result['description']}")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
