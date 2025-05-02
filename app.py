import streamlit as st
from recommender import get_recommendations
from preprocess import load_data

st.title("🎶 Music Recommender System (Content-Based)")

df, cosine_sim = load_data()

user_input = st.text_input("Enter a song title:")

if st.button("Recommend"):
    if user_input:
        results = get_recommendations(user_input, df, cosine_sim)
        if isinstance(results, str):
            st.warning(results)
        else:
            st.subheader("Recommended Songs:")
            for track, artist in results:
                st.write(f"🎧 {track} by {artist}")
    else:
        st.warning("Please enter a song title.")
