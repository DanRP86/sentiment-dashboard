import nltk
nltk.download('punkt')
nltk.download('brown')

import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nrclex import NRCLex
from langdetect import detect, LangDetectException

# Initialize sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment using VADER
def analyze_vader(text):
    scores = vader_analyzer.polarity_scores(text)
    return scores

# Function to analyze sentiment using TextBlob
def analyze_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

# Function to analyze emotions using NRC Emotion Lexicon
def analyze_nrc(text):
    emotion = NRCLex(text)
    emotions = emotion.raw_emotion_scores
    total = sum(emotions.values())
    normalized_emotions = {k: (v / total) * 100 for k, v in emotions.items()}
    return normalized_emotions

# Streamlit app
st.title("Sentiment Analysis Dashboard")

# Sample input text
sample_text = "This is an amazing tool for quickly analyze emotions from text. I love using it!"

# Input text box
text_input = st.text_area("Enter text to analyze:", sample_text, height=200)

if text_input:
    try:
        # Detect language
        language = detect(text_input)
        if language != 'en':
            st.warning("The input text is not in English. The analysis may not be accurate.")

        # Analyze text using VADER
        vader_scores = analyze_vader(text_input)

        # Analyze text using TextBlob
        textblob_polarity, textblob_subjectivity = analyze_textblob(text_input)

        # Analyze text using NRC Emotion Lexicon
        nrc_emotions = analyze_nrc(text_input)

        # Combine results into a DataFrame
        data = {
            "Metric": ["VADER Positive", "VADER Negative", "VADER Neutral", "VADER Compound", "TextBlob Polarity",
                       "TextBlob Subjectivity"] + list(nrc_emotions.keys()),
            "Score": [vader_scores["pos"] * 100, vader_scores["neg"] * 100, vader_scores["neu"] * 100,
                      vader_scores["compound"] * 100, textblob_polarity * 100, textblob_subjectivity * 100] + list(
                nrc_emotions.values())
        }
        df = pd.DataFrame(data)

        # Display results in a table
        st.write("### Sentiment Dashboard")
        st.dataframe(df)

        # Define color mapping for the bar chart
        color_map = {
            "VADER Positive": "green",
            "VADER Negative": "red",
            "VADER Neutral": "blue",
            "VADER Compound": "purple",
            "TextBlob Polarity": "green",
            "TextBlob Subjectivity": "blue"
        }
        for emotion in nrc_emotions.keys():
            if emotion in ["positive", "joy", "trust", "anticipation", "surprise"]:
                color_map[emotion] = "green"
            elif emotion in ["negative", "anger", "disgust", "fear", "sadness"]:
                color_map[emotion] = "red"
            else:
                color_map[emotion] = "blue"

        # Display results in a color-coded bar chart
        st.write("### Sentiment Analysis Visualization")
        fig = px.bar(df, x="Metric", y="Score", title="Sentiment Analysis Scores", labels={"Score": "Percentage"},
                     color="Metric", color_discrete_map=color_map)
        st.plotly_chart(fig)

        # Add text before download link
        st.write("Created by Daniel Rubio")

        # Download results as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download results as CSV", data=csv, file_name='sentiment_analysis_results.csv', mime='text/csv')

    except LangDetectException:
        st.error("Could not detect the language of the input text. Please try again with different text.")


#Put in the console: streamlit run text_sentiment.py
