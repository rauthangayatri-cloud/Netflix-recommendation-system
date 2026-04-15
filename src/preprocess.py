"""
Data loading and preprocessing module.
"""
import pandas as pd
import numpy as np

def load_data():
    """Load movies and ratings from CSV files."""
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    return movies, ratings

def create_user_matrix(ratings):
    """Create a user-movie pivot matrix."""
    matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    )
    return matrix.fillna(0)

def get_movie_stats(ratings, movies):
    """Return average rating and total ratings per movie."""
    stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    stats['avg_rating'] = stats['avg_rating'].round(2)
    return movies.merge(stats, on='movieId', how='left')