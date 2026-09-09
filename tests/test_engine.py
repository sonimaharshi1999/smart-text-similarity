# Author: Maharshi Soni | License: MIT
"""Tests for the similarity engine."""

import pytest
from similarity_engine import SimilarityEngine


@pytest.fixture
def engine():
    return SimilarityEngine()


def test_identical_texts(engine):
    result = engine.compare("hello world", "hello world")
    assert result["combined"] > 0.9


def test_completely_different(engine):
    result = engine.compare("the cat sat on the mat", "quantum physics explores subatomic particles")
    assert result["combined"] < 0.3


def test_similar_texts(engine):
    result = engine.compare(
        "Machine learning is a subset of artificial intelligence",
        "AI includes machine learning as a subfield",
    )
    assert result["combined"] > 0.3


def test_jaccard_identical():
    score = SimilarityEngine.jaccard_similarity("hello world", "hello world")
    assert score == 1.0


def test_jaccard_no_overlap():
    score = SimilarityEngine.jaccard_similarity("cat dog", "fish bird")
    assert score == 0.0


def test_empty_texts():
    score = SimilarityEngine.jaccard_similarity("", "")
    assert score == 1.0


def test_result_keys(engine):
    result = engine.compare("text one", "text two")
    assert "tfidf_cosine" in result
    assert "jaccard" in result
    assert "combined" in result


def test_scores_in_range(engine):
    result = engine.compare("some text here", "other text there")
    for key in ["tfidf_cosine", "jaccard", "combined"]:
        assert 0.0 <= result[key] <= 1.0
