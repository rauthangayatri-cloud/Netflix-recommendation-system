"""
Main Recommender module.
"""
import pandas as pd
from src.preprocess import load_data, create_user_matrix, get_movie_stats
from src.collaborative import recommend_collaborative
from src.content_based import build_tfidf_matrix, build_similarity_matrix, recommend_content_based

class MovieRecommender:
    """Unified recommendation engine."""

    def __init__(self):
        print("Loading data...")
        self.movies, self.ratings = load_data()
        self.movie_stats = get_movie_stats(self.ratings, self.movies)

        print("Building user matrix...")
        self.user_matrix = create_user_matrix(self.ratings)

        print("Building content model...")
        tfidf_matrix, _ = build_tfidf_matrix(self.movies)
        self.similarity_matrix = build_similarity_matrix(tfidf_matrix)
        print("Ready!")

    def get_similar_movies(self, movie_title, n=10):
        """Content-based: movies similar to the given title."""
        return recommend_content_based(
            movie_title, self.movies, self.similarity_matrix, n
        )

    def get_user_recommendations(self, user_id, n=10):
        """Collaborative: recommendations for a specific user ID."""
        return recommend_collaborative(
            user_id, self.user_matrix, self.movies, n
        )

    def get_popular_movies(self, n=10, min_ratings=50):
        """Return top-rated popular movies."""
        popular = self.movie_stats[
            self.movie_stats['num_ratings'] >= min_ratings
        ].sort_values('avg_rating', ascending=False)
        return popular[['title', 'genres', 'avg_rating', 'num_ratings']].head(n)