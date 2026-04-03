import pytest

from cli.index import InvertedIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Build / indexing
# ---------------------------------------------------------------------------

def test_build_indexes_tokens(index):
    assert 1 in index.get_documents("brave")


def test_build_does_not_index_other_doc(index):
    assert 2 not in index.get_documents("brave")


def test_build_indexes_stemmed_tokens(index):
    # "fights" stems to "fight"
    assert 1 in index.get_documents("fight")
    assert index.get_documents("fights") == []


def test_duplicate_document_is_ignored(stopwords):
    idx = InvertedIndex()
    movie = {"id": 1, "title": "Brave Hero", "description": "A brave warrior fights."}
    idx._add_document(1, movie, stopwords)
    idx._add_document(1, movie, stopwords)
    assert idx.docmap[1] == movie


def test_get_documents_missing_term_returns_empty(index):
    assert index.get_documents("nonexistent") == []


# ---------------------------------------------------------------------------
# Term frequency (TF)
# ---------------------------------------------------------------------------

def test_tf_counts_repeated_terms(stopwords):
    idx = InvertedIndex()
    movie = {"id": 1, "title": "Bear Bear", "description": "A bear story."}
    idx.build([movie], stopwords)
    # "bear" appears 3 times — Counter must preserve duplicates
    assert idx.term_frequencies[1]["bear"] == 3


# ---------------------------------------------------------------------------
# Inverse document frequency (IDF)
# ---------------------------------------------------------------------------

def test_idf_equal_for_same_doc_frequency(index):
    # "brave" and "space" each appear in exactly 1 of 2 docs → equal IDF
    idf_brave = index.inverse_doc_frequencies.get("brave", 0)
    idf_space = index.inverse_doc_frequencies.get("space", 0)
    assert idf_brave == idf_space


def test_idf_common_term_lower_than_rare(stopwords):
    movies = [
        {"id": 1, "title": "Hero", "description": "A warrior fights dragons."},
        {"id": 2, "title": "Dragon Quest", "description": "A journey with dragons."},
        {"id": 3, "title": "Space", "description": "A journey through space."},
    ]
    idx = InvertedIndex()
    idx.build(movies, stopwords)
    # "dragon" in 2 docs, "space" in 1 doc → dragon IDF < space IDF
    assert idx.inverse_doc_frequencies["dragon"] < idx.inverse_doc_frequencies["space"]


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def test_get_tf_idf_returns_zero_for_absent_term(index, stopwords):
    # "brave" is not in doc 2
    assert index.get_tf_idf(2, "brave", stopwords) == 0.0


def test_get_tf_idf_positive_for_present_term(index, stopwords):
    assert index.get_tf_idf(1, "brave", stopwords) > 0.0


def test_get_tf_idf_invalid_doc_raises(index, stopwords):
    with pytest.raises(KeyError, match="99999"):
        index.get_tf_idf(99999, "brave", stopwords)


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(index, tmp_path):
    index.save(tmp_path)
    loaded = InvertedIndex.load(tmp_path)
    assert loaded.get_documents("brave") == index.get_documents("brave")
    assert loaded.docmap == index.docmap


def test_load_restores_counter_type(index, tmp_path, stopwords):
    index.save(tmp_path)
    loaded = InvertedIndex.load(tmp_path)
    # Counter returns 0 for missing keys; plain dict raises KeyError
    assert loaded.term_frequencies[1]["nonexistent_token"] == 0


