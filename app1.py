import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('LR_model.jb')

# Custom CSS for Background & Styling
st.markdown("""
    <style>
        /* Background color */
        .stApp {
            background-color: #f4f4f8;
        }

        /* Title text */
        .title {
            color: #ff6347;
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }

        /* Subtext */
        .subtext {
            color: #555;
            font-size: 18px;
            text-align: center;
        }

        /* Custom button */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px;
            border: none;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<h1 class="title">FAKE NEWS DETECTOR🔍</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Enter a news article below to check whether it is <b>Fake</b> or <b>Real</b>.</p>', unsafe_allow_html=True)

# User Input
news_input = st.text_area("News Article:", "")

# Button to Predict
if st.button('Check News'):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("THE NEWS IS REAL!")
        else:
            st.error("THE NEWS IS FAKE!❌")
    else:
        st.warning("⚠️PLEASE ENTER SOME TEXT TO ANALYZE.")
