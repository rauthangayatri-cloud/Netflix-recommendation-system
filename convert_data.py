import pandas as pd

# Convert ratings
ratings = pd.read_csv(
    'data/ml-100k/u.data',
    sep='\t',
    names=['userId', 'movieId', 'rating', 'timestamp']
)
ratings.drop('timestamp', axis=1, inplace=True)
ratings.to_csv('data/ratings.csv', index=False)
print(f"Ratings saved: {len(ratings)} rows")

# Convert movies
movies = pd.read_csv(
    'data/ml-100k/u.item',
    sep='|',
    encoding='latin-1',
    usecols=[0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    names=['movieId', 'title', 'unknown', 'Action', 'Adventure', 'Animation',
           'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
           'Fantasy', 'FilmNoir', 'Horror', 'Musical', 'Mystery',
           'Romance', 'SciFi', 'Thriller', 'War', 'Western']
)

genre_cols = movies.columns[2:]
movies['genres'] = movies[genre_cols].apply(
    lambda row: ' '.join([col for col, val in row.items() if val == 1]),
    axis=1
)
movies = movies[['movieId', 'title', 'genres']]
movies.to_csv('data/movies.csv', index=False)
print(f"Movies saved: {len(movies)} rows")
print("Done! Check data/ folder.")