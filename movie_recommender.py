from flask import Flask, render_template, request
import pandas as pd
import sqlite3  # ✅ THIS LINE IS NEEDED
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and process data
movies = pd.read_csv('tmdb_5000_movies.csv')
movies = movies[['title', 'overview']]
movies.dropna(inplace=True)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation logic
def recommend_movie(title, num_recommendations=5):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie']
    results = recommend_movie(movie_title)
    return render_template('result.html', movie=movie_title, recommendations=results)

@app.route("/filter", methods=["POST"])
def filter_movies():
    choice = request.form["filter"]
    genre = request.form.get("genre", "").strip().lower()
    min_rating = request.form.get("min_rating")
    release_year = request.form.get("release_year")

    query = "SELECT title, release_date, vote_average FROM Movies WHERE 1=1"
    params = []

    if choice == "genre" and genre:
        query += " AND lower(genres) LIKE ?"
        params.append(f"%{genre}%")

    if min_rating:
        query += " AND vote_average >= ?"
        params.append(float(min_rating))

    if release_year:
        query += " AND release_date >= ?"
        params.append(f"{release_year}-01-01")

    if choice == "latest":
        query += " ORDER BY release_date DESC LIMIT 10"
    elif choice == "top":
        query += " ORDER BY popularity DESC LIMIT 10"
    else:
        query += " ORDER BY vote_average DESC LIMIT 10"

    conn = sqlite3.connect("movies.db")
    cursor = conn.cursor()
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    heading = "🎬 Filtered Results"
    return render_template("filter_results.html", heading=heading, movies=results)

if __name__ == '__main__':
    app.run(debug=True)
