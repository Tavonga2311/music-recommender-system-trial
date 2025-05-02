from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(title, df, tfidf_matrix):
    # Find the index of the song title
    indices = df[df['track_name'].str.lower() == title.lower()].index
    if len(indices) == 0:
        return "Song not found in dataset."
    
    idx = indices[0]
    
    # Reshape the vector to 2D (1, n_features)
    song_vector = tfidf_matrix[idx].reshape(1, -1)
    
    # Compute the cosine similarity for just this song against the others
    sim_scores = cosine_similarity(song_vector, tfidf_matrix)
    
    # Get top 5 similar songs (excluding the song itself)
    sim_scores = list(enumerate(sim_scores[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    
    recommended = []
    for i in sim_scores:
        track = df.iloc[i[0]]['track_name']
        artist = df.iloc[i[0]]['artist_name']
        recommended.append((track, artist))
        
    return recommended
