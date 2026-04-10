from pathlib import Path
import json


def load_movies(filepath: Path) -> list[dict]:
    with open(filepath) as f:
        return json.load(f)["movies"]


def load_stopwords(filepath: Path) -> set[str]:
    with open(filepath) as f:
        return set(f.read().splitlines())
