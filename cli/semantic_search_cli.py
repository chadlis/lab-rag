#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from lib.loader import load_movies
from lib.semantic_search import embed_text, SemanticSearch, ChunkedSemanticSearch, chunk_text, semantic_chunk_text

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
    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=40)
    subparsers.add_parser("embed_chunks", help="Build or load chunked embeddings")
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies using chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="the search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="number of results to return")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)
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
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            chunks = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
        case "embed_chunks":
            movies = load_movies(DATA_DIR / MOVIES_FILENAME)
            chunked_search = ChunkedSemanticSearch()
            embeddings = chunked_search.load_or_create_chunk_embeddings(movies, CACHE_DIR)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            movies = load_movies(DATA_DIR / MOVIES_FILENAME)
            chunked_search = ChunkedSemanticSearch()
            chunked_search.load_or_create_chunk_embeddings(movies, CACHE_DIR)
            results = chunked_search.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
