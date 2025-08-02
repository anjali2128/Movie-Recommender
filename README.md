# 🎬 Movie Recommendation System

A personalized movie recommendation web app built using Python, Flask, and Natural Language Processing. Suggests similar movies based on the input title using content-based filtering and cosine similarity.


---

## 🚀 Features

- 🔍 Enter a movie name and get similar movie suggestions instantly
- 📚 Uses TF-IDF vectorization and cosine similarity for content-based filtering
- 🧠 Built with machine learning techniques and NLP on movie metadata
- 🧼 Clean and minimal UI with real-time responses
- 📦 Lightweight Flask backend, easy to deploy anywhere

---

## 🛠️ Tech Stack

**Frontend:**  
HTML5, CSS3, Bootstrap

**Backend:**  
Python, Flask

**Machine Learning / NLP:**  
Pandas, Scikit-learn, TF-IDF, Cosine Similarity

**Dataset:**  
TMDB 5000 Movies Dataset

---

## 📁 Project Structure

movie/
├── static/ # CSS, JS, assets
├── templates/ # HTML templates (home, result)
├── tmdb_5000_movies.csv # Movie data
├── movie_recommender.py # Flask application
├── recommendation.py # ML logic for similarity
└── README.md 
---

## 💻 How to Run Locally

### 🔧 Step-by-Step

1. Clone the repository:

```bash
git clone https://github.com/anjali2128/movie.git
cd movie
```
---
##📈 Future Enhancements

🔗 Integrate TMDB API to fetch real posters, ratings, trailers
👥 Add collaborative filtering (user-based)
🌍 Deploy using Render, Railway, or Heroku
🪄 Add search suggestions/autocomplete
