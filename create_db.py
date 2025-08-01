import pandas as pd
import sqlite3

# Load your movie CSV file
movies = pd.read_csv("tmdb_5000_movies.csv")

# Create the SQLite DB and table
conn = sqlite3.connect("movies.db")
movies.to_sql("Movies", conn, if_exists="replace", index=False)
conn.close()

print("✅ movies.db created")
