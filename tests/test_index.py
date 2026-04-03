import pytest

from cli.index import InvertedIndex


@pytest.fixture
def movies():
    return [
        {"id": 1, "title": "Brave Hero", "description": "A brave warrior fights."},
        {"id": 2, "title": "Space Odyssey", "description": "A journey through space."},
    ]


@pytest.fixture
def stopwords():
    return {"a", "an", "the", "and", "is", "about"}


@pytest.fixture
def index(movies, stopwords):
    idx = InvertedIndex()
    idx.build(movies, stopwords)
    return idx


def test_build_indexes_tokens(index):
    assert 1 in index.get_documents("brave")


def test_build_does_not_index_other_doc(index):
    assert 2 not in index.get_documents("brave")


def test_get_documents_missing_term_returns_empty(index):
    assert index.get_documents("nonexistent") == []


def test_duplicate_document_is_ignored(stopwords):
    idx = InvertedIndex()
    movie = {"id": 1, "title": "Brave Hero", "description": "A brave warrior fights."}
    idx._add_document(1, movie, stopwords)
    idx._add_document(1, movie, stopwords)
    assert idx.docmap[1] == movie


def test_index_stores_stemmed_tokens(index):
    # "fights" is stemmed to "fight" during build
    assert 1 in index.get_documents("fight")
    assert index.get_documents("fights") == []


def test_save_and_load_round_trip(index, tmp_path):
    index.save(tmp_path)
    loaded = InvertedIndex.load(tmp_path)
    assert loaded.get_documents("brave") == index.get_documents("brave")
    assert loaded.docmap == index.docmap
