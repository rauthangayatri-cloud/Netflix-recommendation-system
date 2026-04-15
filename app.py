"""
Netflix-Style Recommendation System
"""
import streamlit as st
import pandas as pd
from src.recommender import MovieRecommender

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    return MovieRecommender()

recommender = load_recommender()

st.sidebar.title("Settings")
mode = st.sidebar.radio(
    "Recommendation Mode",
    ["Find similar movies", "Recommend for a user", "Popular movies"]
)
num_results = st.sidebar.slider("Number of results", 5, 20, 10)
st.sidebar.markdown("---")
st.sidebar.markdown("**About this app**")
st.sidebar.markdown(
    "Uses MovieLens 100K dataset. "
    "Content-based filtering uses TF-IDF on genres. "
    "Collaborative filtering uses cosine similarity on user ratings."
)

st.title("Movie Recommendation System")
st.markdown("A Netflix-style recommender built with Python and Scikit-learn")
st.markdown("---")

if mode == "Find similar movies":
    st.header("Content-Based Filtering")
    st.write("Select a movie and get similar ones based on genre.")
    movie_list = sorted(recommender.movies['title'].tolist())
    selected = st.selectbox("Choose a movie:", movie_list, index=0)
    if st.button("Get Similar Movies", type="primary"):
        results = recommender.get_similar_movies(selected, n=num_results)
        if results.empty:
            st.warning("No similar movies found. Try another title.")
        else:
            st.success(f"Top {len(results)} movies similar to **{selected}**")
            st.dataframe(results[['title', 'genres']].reset_index(drop=True),
                        use_container_width=True, hide_index=True)

elif mode == "Recommend for a user":
    st.header("Collaborative Filtering")
    st.write("Enter a User ID (1–943) to get personalized recommendations.")
    user_id = st.number_input("User ID:", min_value=1, max_value=943, value=1, step=1)
    if st.button("Get Recommendations", type="primary"):
        results = recommender.get_user_recommendations(int(user_id), n=num_results)
        if results.empty:
            st.warning("Could not generate recommendations for this user.")
        else:
            st.success(f"Top {len(results)} picks for User #{user_id}")
            st.dataframe(results[['title', 'genres']].reset_index(drop=True),
                        use_container_width=True, hide_index=True)
        user_history = (
            recommender.ratings[recommender.ratings['userId'] == user_id]
            .merge(recommender.movies, on='movieId')
            .sort_values('rating', ascending=False)
            .head(5)
        )
        st.subheader(f"User #{user_id}'s top-rated movies")
        st.dataframe(user_history[['title', 'rating']].reset_index(drop=True),
                    use_container_width=True, hide_index=True)

elif mode == "Popular movies":
    st.header("Top Rated Movies")
    st.write("Most popular movies with at least 50 ratings.")
    results = recommender.get_popular_movies(n=num_results)
    st.dataframe(results.reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Dataset: MovieLens 100K by GroupLens Research | Built with Streamlit, Pandas, NumPy, Scikit-learn")