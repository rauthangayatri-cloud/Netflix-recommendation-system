"""
Collaborative Filtering module.
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_user_similarity(matrix):
    """Compute cosine similarity between all users."""
    sim = cosine_similarity(matrix)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

def get_similar_users(user_id, sim_df, n=10):
    """Return top-n most similar users to the given user."""
    if user_id not in sim_df.index:
        return pd.Series(dtype=float)
    similar = sim_df[user_id].drop(user_id).sort_values(ascending=False)
    return similar.head(n)

def recommend_collaborative(user_id, matrix, movies_df, n=10):
    """Recommend movies based on collaborative filtering."""
    sim_df = get_user_similarity(matrix)
    similar_users = get_similar_users(user_id, sim_df, n=20)

    if similar_users.empty:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])

    user_rated = set(matrix.loc[user_id][matrix.loc[user_id] > 0].index)

    score_map = {}
    for uid, similarity in similar_users.items():
        if uid not in matrix.index:
            continue
        user_movies = matrix.loc[uid]
        for movie_id, rating in user_movies[user_movies > 0].items():
            if movie_id not in user_rated:
                if movie_id not in score_map:
                    score_map[movie_id] = 0
                score_map[movie_id] += similarity * rating

    if not score_map:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])

    top_movies = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:n]
    top_ids = [mid for mid, _ in top_movies]
    result = movies_df[movies_df['movieId'].isin(top_ids)].copy()
    return result.reset_index(drop=True)