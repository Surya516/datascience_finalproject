# app.py

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.datasets import imdb 
import numpy as np

# Parameters
max_features = 20000  # Number of words to consider as features
maxlen = 200          # Cut reviews after 200 words

# Load IMDb word index
word_index = imdb.get_word_index()
index_from = 3  # Indices reserved: 0 (padding), 1 (start), 2 (OOV)

# Adjust word indices
word_index = {k: (v + index_from) for k, v in word_index.items()}
# Add special tokens to word index
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
reverse_word_index = {v: k for k, v in word_index.items()}

# Load the trained model
@st.cache_data
def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model

model = load_model()

# Sentiment Analysis Function
def analyze_sentiment(review):
    """Analyze sentiment of a review using the trained model."""
    # Preprocess the review
    words = review.lower().split()
    sequence = []
    for word in words:
        index = word_index.get(word, 2)  # Use index 2 for unknown words
        if index >= max_features:
            index = 2  # Map words outside max_features to OOV token
        sequence.append(index)
    # Pad the sequence
    padded_sequence = pad_sequences([sequence], maxlen=maxlen)
    # Predict sentiment
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    # Return sentiment score
    return 1 if prediction >= 0.5 else -1

# Streamlit App
st.title("🎬 Sentiment-Based Movie Rankings")

# Initialize Movie Data in Session State
if 'movies' not in st.session_state:
    st.session_state.movies = [
        {"title": "Inception", "reviews": [], "score": 0, "image": "images/inception.jpg"},
        {"title": "The Dark Knight", "reviews": [], "score": 0, "image": "images/the_dark_knight.jpg"},
        {"title": "Interstellar", "reviews": [], "score": 0, "image": "images/interstellar.jpg"},
        {"title": "Joker", "reviews": [], "score": 0, "image": "images/joker.jpg"},
        {"title": "Parasite", "reviews": [], "score": 0, "image": "images/parasite.jpg"},
        {"title": "Titanic", "reviews": [], "score": 0, "image": "images/titanic.jpg"},
        {"title": "The Matrix", "reviews": [], "score": 0, "image": "images/the_matrix.jpg"},
        {"title": "Avengers: Endgame", "reviews": [], "score": 0, "image": "images/avengers_endgame.jpg"},
        {"title": "The Lion King", "reviews": [], "score": 0, "image": "images/the_lion_king.jpg"},
        {"title": "Frozen", "reviews": [], "score": 0, "image": "images/frozen.jpg"},
    ]

# Optionally, display model test accuracy
if 'test_evaluated' not in st.session_state:
    st.session_state.test_evaluated = False

if not st.session_state.test_evaluated:
    # Load IMDb test data for evaluation
    (_, _), (x_test_eval, y_test_eval) = imdb.load_data(num_words=max_features)
    x_test_eval = pad_sequences(x_test_eval, maxlen=maxlen)
    test_loss, test_accuracy = model.evaluate(x_test_eval, y_test_eval, verbose=0)
    st.session_state.test_evaluated = True

# Display Movies and Rankings
def display_movies():
    """Display the current ranking of movies with images."""
    sorted_movies = sorted(st.session_state.movies, key=lambda x: x["score"], reverse=True)
    st.subheader("🏆 Current Movie Rankings")
    for idx, movie in enumerate(sorted_movies):
        cols = st.columns([3, 9])  # Adjust column widths as needed
        with cols[0]:
            if os.path.exists(movie["image"]):
                st.image(movie["image"], width=100)
            else:
                st.write("🎥 No Image Available")
        with cols[1]:
            st.markdown(f"**{idx + 1}. {movie['title']}**")
            st.write(f"**Sentiment Score:** {movie['score']}")
            if movie["reviews"]:
                st.write(f"**Reviews:** {', '.join(movie['reviews'])}")
            else:
                st.write("**Reviews:** None")
        st.write("---")

flag = 0

# Add a Review Using a Form
st.subheader("📝 Add a Review")
with st.form("review_form", clear_on_submit=True):
    movie_choice = st.selectbox("Select a Movie", [movie["title"] for movie in st.session_state.movies])
    review = st.text_area("Enter your review")
    submit = st.form_submit_button("Submit Review")

    if submit:
        flag = 1
        if review.strip():
            # Find the movie in session state
            try:
                movie_index = next(i for i, m in enumerate(st.session_state.movies) if m["title"] == movie_choice)
            except StopIteration:
                st.error("Selected movie not found. Please try again.")
                st.stop()
            # Analyze sentiment
            score = analyze_sentiment(review)
            # Update the movie's data
            st.session_state.movies[movie_index]["reviews"].append(review)
            st.session_state.movies[movie_index]["score"] += score
            # Provide feedback
            if score == 1:
                st.success("👍 Positive review added! Rankings updated.")
            else:
                st.warning("👎 Negative review added! Rankings updated.")
            # Rerun to update the display
            display_movies()            
        else:
            st.error("❌ Review cannot be empty. Please try again.")
            display_movies()     
    # Display the current movie rankings
    if flag:
        pass
    else:
        display_movies()       





