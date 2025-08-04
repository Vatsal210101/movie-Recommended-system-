from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import pickle
import os

app = Flask(__name__)
CORS(app)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Global variables
movies_df = None
similarity_matrix = None
ps = PorterStemmer()

def convert(obj):
    """Convert JSON string to list of names"""
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except:
        return []
    return L

def convert3(obj):
    """Convert cast JSON to list of top 3 names"""
    L = []
    c = 0
    try:
        for i in ast.literal_eval(obj):
            if c != 3:
                L.append(i['name'])
                c += 1
            else:
                break
    except:
        return []
    return L

def fd(obj):
    """Extract director name from crew JSON"""
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except:
        return []
    return L

def stem(text):
    """Apply stemming to text"""
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def load_and_process_data():
    """Load and process the movie data"""
    global movies_df, similarity_matrix
    
    print("Loading movie data...")
    
    # Load the datasets
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    # Merge datasets
    movies = movies.merge(credits, on='title')
    
    # Select relevant columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Remove null values
    movies.dropna(inplace=True)
    
    # Process genres
    movies['genres'] = movies['genres'].apply(convert)
    
    # Process keywords
    movies['keywords'] = movies['keywords'].apply(convert)
    
    # Process cast (top 3)
    movies['cast'] = movies['cast'].apply(convert3)
    
    # Process crew (director only)
    movies['crew'] = movies['crew'].apply(fd)
    
    # Process overview
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    # Remove spaces from all lists
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    # Create tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create new dataframe
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    
    # Convert tags to string
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    
    # Convert to lowercase
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    # Apply stemming
    new_df['tags'] = new_df['tags'].apply(stem)
    
    movies_df = new_df
    
    print("Creating similarity matrix...")
    
    # Create CountVectorizer
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vectors)
    
    print("Data processing complete!")

@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Get list of all movies"""
    if movies_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    movies_list = movies_df['title'].tolist()
    return jsonify({'movies': movies_list})

@app.route('/api/recommend', methods=['POST'])
def recommend_movies():
    """Get movie recommendations"""
    if movies_df is None or similarity_matrix is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    data = request.get_json()
    movie_title = data.get('movie_title', '').strip()
    
    if not movie_title:
        return jsonify({'error': 'Movie title is required'}), 400
    
    try:
        # Find movie index
        movie_index = movies_df[movies_df['title'].str.lower() == movie_title.lower()].index[0]
        
        # Get similarity scores
        distances = similarity_matrix[movie_index]
        
        # Get top 5 similar movies (excluding the movie itself)
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        # Get recommended movies
        recommended_movies = []
        for i in movies_list:
            movie_data = {
                'title': movies_df.iloc[i[0]]['title'],
                'movie_id': int(movies_df.iloc[i[0]]['movie_id']),
                'similarity_score': float(i[1])
            }
            recommended_movies.append(movie_data)
        
        return jsonify({
            'selected_movie': movie_title,
            'recommendations': recommended_movies
        })
        
    except IndexError:
        return jsonify({'error': f'Movie "{movie_title}" not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search movies by title"""
    if movies_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({'movies': []})
    
    # Filter movies that contain the query
    matching_movies = movies_df[movies_df['title'].str.lower().str.contains(query)]
    
    movies_list = matching_movies['title'].tolist()[:10]  # Limit to 10 results
    return jsonify({'movies': movies_list})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'data_loaded': movies_df is not None})

if __name__ == '__main__':
    # Load data when starting the server
    load_and_process_data()
    
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000) 