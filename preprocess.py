import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    # Load your dataset (assumes it's named "tracks.csv")
    df = pd.read_csv("tracks1.csv", encoding='ISO-8859-1')

    # Use only relevant text features for similarity
    df = df[['track_name', 'artist_name', 'genre']].dropna()

    # Create combined feature column
    df['combined'] = df['track_name'] + ' ' + df['artist_name'] + ' ' + df['genre']

    # Vectorize the combined text
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined'])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, cosine_sim
