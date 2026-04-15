🎬 Netflix-Style Movie Recommendation System
A machine learning-powered movie recommendation system built with Python and Scikit-learn, inspired by Netflix's recommendation engine. Uses the MovieLens 100K dataset to suggest movies based on content similarity and user behavior.

📸 Screenshots
<img width="1920" height="1020" alt="Screenshot 2026-04-15 202227" src="https://github.com/user-attachments/assets/72c9dc64-1f4a-4aa2-9c4f-44c1b184d294" />
<img width="1920" height="1020" alt="Screenshot 2026-04-15 202140" src="https://github.com/user-attachments/assets/de612484-5942-46a7-b2de-ad2c1e2bc4a1" />


🎯 Objective
To build a recommendation system that suggests relevant movies to users using two approaches:

Content-Based Filtering — recommends movies similar to a selected movie based on genre
Collaborative Filtering — recommends movies based on what similar users have liked
Popular Movies — shows trending movies ranked by average rating and view count


✨ Features

🔍 Find Similar Movies — select any movie and get top N similar movies by genre
👤 User-Based Recommendations — enter a User ID and get personalized suggestions
🔥 Popular Movies — browse highest-rated and most-watched movies
🎛️ Adjustable Results — slider to control number of recommendations (1–20)
📊 Clean Data Table — results shown with title, genre, and rating info
🌐 Interactive Web UI — built with Streamlit, runs in the browser


## 🛠️ Tools & Technologies

| Technology       | Purpose                                  |
|----------------|------------------------------------------|
| Python 3.x      | Core programming language                |
| Streamlit       | Web application framework                |
| Pandas          | Data manipulation and analysis           |
| NumPy           | Numerical computations                   |
| Scikit-learn    | TF-IDF vectorization, similarity         |
| MovieLens 100K  | Movie ratings dataset                    |
| Git & GitHub    | Version control                          |


📁 Project Structure
Netflix_mang_sys/
│
├── data/
│   ├── movies.csv          # Movie titles and genres
│   ├── ratings.csv         # User ratings data
│   └── ml-100k/            # Raw MovieLens 100K dataset
│
├── src/
│   ├── preprocess.py       # Data loading and cleaning
│   ├── content_based.py    # TF-IDF + cosine similarity logic
│   ├── collaborative.py    # User-based collaborative filtering
│   └── recommender.py      # Main recommender interface
│
├── app.py                  # Streamlit web application
├── convert_data.py         # Dataset conversion utility
├── requirements.txt        # Python dependencies
└── README.md

⚙️ How It Works
Content-Based Filtering

Movie genres are converted into TF-IDF feature vectors
Cosine similarity is computed between all movie vectors
Given a selected movie, the top N most similar movies are returned

Collaborative Filtering

A user-item rating matrix is built from the ratings dataset
Cosine similarity is computed between users
Movies liked by similar users (but not yet seen) are recommended

Popular Movies

Movies are ranked by average rating × number of ratings
Top N movies are returned as trending/popular picks


🚀 Getting Started
Prerequisites

Python 3.8 or higher
pip

Installation
bash# Clone the repository
git clone https://github.com/rauthangayatri-cloud/Netflix-recommendation-system.git
cd Netflix-recommendation-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
Run the App
bashstreamlit run app.py
Then open your browser and go to: http://localhost:8501

📊 Dataset
This project uses the MovieLens 100K dataset by GroupLens Research.

100,000 ratings from 943 users on 1,682 movies
Ratings scale: 1–5 stars
Source: grouplens.org/datasets/movielens/100k


📦 Requirements
streamlit
pandas
numpy
scikit-learn
Install all with:
bashpip install -r requirements.txt

🔮 Future Improvements

 Add movie posters using TMDB API
 Implement matrix factorization (SVD) for better collaborative filtering
 Add search bar to find movies by name
 Deploy to Azure App Service or Streamlit Cloud
 Add user login and save preference history


👩‍💻 Author
Gayatri 
GitHub: @rauthangayatri-cloud

📄 License
This project is for educational purposes as part of a cloud computing course at SGT University.
