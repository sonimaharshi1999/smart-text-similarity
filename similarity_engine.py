# Author: Maharshi Soni | License: MIT
"""Core similarity engine with multiple algorithms."""

import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    def __init__(self, tfidf_weight: float = 0.6, jaccard_weight: float = 0.4):
        self.tfidf_weight = tfidf_weight
        self.jaccard_weight = jaccard_weight
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r'\b\w+\b', text.lower()))

    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        tokens1 = SimilarityEngine._tokenize(text1)
        tokens2 = SimilarityEngine._tokenize(text2)
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def tfidf_cosine_similarity(self, text1: str, text2: str) -> float:
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])

    def compare(self, text1: str, text2: str) -> dict:
        tfidf_score = self.tfidf_cosine_similarity(text1, text2)
        jaccard_score = self.jaccard_similarity(text1, text2)
        combined = (self.tfidf_weight * tfidf_score) + (self.jaccard_weight * jaccard_score)

        return {
            "tfidf_cosine": round(tfidf_score, 4),
            "jaccard": round(jaccard_score, 4),
            "combined": round(combined, 4),
        }

    def batch_compare(self, csv_path: str, threshold: float = 0.5) -> list[dict]:
        df = pd.read_csv(csv_path)
        if "text" not in df.columns:
            raise ValueError("CSV must have a 'text' column")

        texts = df["text"].tolist()
        results = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                score = self.compare(texts[i], texts[j])
                if score["combined"] >= threshold:
                    results.append({
                        "text1": texts[i],
                        "text2": texts[j],
                        "score": score["combined"],
                        "details": score,
                    })

        return sorted(results, key=lambda x: -x["score"])
