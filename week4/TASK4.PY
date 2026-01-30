# ================================
# FASTAPI MOVIE RECOMMENDATION SYSTEM (BEGINNER FRIENDLY)
# ================================
# STEP 1: INSTALL REQUIRED PACKAGES (RUN IN TERMINAL)
# pip install fastapi uvicorn pandas scikit-learn

# ================================
# STEP 2: IMPORT LIBRARIES
# ================================
from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# STEP 3: CREATE FASTAPI APP
# ================================
app = FastAPI(title="Movie Recommendation API")

# ================================
# STEP 4: LOAD DATASET
# ================================
# NOTE: Keep tmdb_5000_movies.csv in the SAME folder as this file
movies = pd.read_csv("tmdb_5000_movies.csv")

# Select only required columns
movies = movies[['title', 'overview', 'genres']]

# Fill missing values
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')

# ================================
# STEP 5: TEXT CLEANING
# ================================
movies['tags'] = movies['overview'] + " " + movies['genres']

# ================================
# STEP 6: VECTORIZE TEXT DATA
# ================================
cv = CountVectorizer(max_features=3000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# ================================
# STEP 7: COSINE SIMILARITY
# ================================
similarity = cosine_similarity(vectors)

# ================================
# STEP 8: RECOMMENDATION LOGIC
# ================================
def recommend_movie(movie_name: str):
    if movie_name not in movies['title'].values:
        return []

    index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movies_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations

# ================================
# STEP 9: API ENDPOINT
# ================================
@app.get("/recommend")
def recommend(movie: str):
    result = recommend_movie(movie)
    return {
        "input_movie": movie,
        "recommended_movies": result
    }

# ================================
# STEP 10: RUN SERVER
# ================================
# Run using command:
# uvicorn main:app --reload
