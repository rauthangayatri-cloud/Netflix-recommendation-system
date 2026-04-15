"""
Content-Based Filtering module.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf_matrix(movies_df):
    """Build a TF-IDF matrix from movie genres."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'].fillna(''))
    return tfidf_matrix, tfidf

def build_similarity_matrix(tfidf_matrix):
    """Compute cosine similarity between all movies."""
    return cosine_similarity(tfidf_matrix)

def recommend_content_based(movie_title, movies_df, similarity_matrix, n=10):
    """Return n movies most similar to the given movie title."""
    matches = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        matches = movies_df[movies_df['title'].str.lower().str.contains(
            movie_title.lower(), na=False
        )]
    if matches.empty:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])

    idx = matches.index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, s in scores if i != idx and s > 0][:n]
    result = movies_df.iloc[top_indices][['movieId', 'title', 'genres']].copy()
    return result.reset_index(drop=True)