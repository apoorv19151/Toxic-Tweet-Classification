from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

def create_bow_vectorizer(max_features=50000, ngram_range=(1, 2), min_df=3):
    return CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )

def create_tfidf_vectorizer(max_features=50000, ngram_range=(1, 2), min_df=3):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )

__all__ = [
    "create_bow_vectorizer",
    "create_tfidf_vectorizer",
]
