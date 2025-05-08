
import nltk
import textblob.download_corpora
import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nrclex import NRCLex
from langdetect import detect, LangDetectException

# Download required corpora
nltk.download('punkt')
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
textblob.download_corpora.download_all()

# Initialize sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_vader(text):
    return vader_analyzer.polarity_scores(text)

def analyze_textblob(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception as e:
        st.error(f"TextBlob error: {e}")
        return 0, 0

def analyze_nrc(text):
    try:
        emotion = NRCLex(text)
        scores = emotion.raw_emotion_scores
        total = sum(scores.values())
        return {k: (v / total) * 100 for k, v in scores.items()} if total > 0 else {}
    except Exception as e:
        st.error(f"NRC error: {e}")
        return {}

# Streamlit UI
st.title("Sentiment Analysis Dashboard")
sample_text = "This is an amazing tool for quickly analyze emotions from text. I love using it!"
text_input = st.text_area("Enter text to analyze:", sample_text, height=200)

if text_input:
    try:
        if detect(text_input) != 'en':
            st.warning("The input text is not in English. The analysis may not be accurate.")
    except LangDetectException:
        st.error("Language detection failed.")

    vader_scores = analyze_vader(text_input)
    polarity, subjectivity = analyze_textblob(text_input)
    nrc_emotions = analyze_nrc(text_input)

    data = {
        "Metric": ["VADER Positive", "VADER Negative", "VADER Neutral", "VADER Compound", "TextBlob Polarity", "TextBlob Subjectivity"] + list(nrc_emotions.keys()),
        "Score": [vader_scores["pos"] * 100, vader_scores["neg"] * 100, vader_scores["neu"] * 100, vader_scores["compound"] * 100, polarity * 100, subjectivity * 100] + list(nrc_emotions.values())
    }
    df = pd.DataFrame(data)

    st.write("### Sentiment Dashboard")
    st.dataframe(df)

    color_map = {
        "VADER Positive": "green", "VADER Negative": "red", "VADER Neutral": "blue", "VADER Compound": "purple",
        "TextBlob Polarity": "green", "TextBlob Subjectivity": "blue"
    }
    for emotion in nrc_emotions:
        if emotion in ["positive", "joy", "trust", "anticipation", "surprise"]:
            color_map[emotion] = "green"
        elif emotion in ["negative", "anger", "disgust", "fear", "sadness"]:
            color_map[emotion] = "red"
        else:
            color_map[emotion] = "blue"

    st.write("### Sentiment Analysis Visualization")
    fig = px.bar(df, x="Metric", y="Score", title="Sentiment Analysis Scores", labels={"Score": "Percentage"}, color="Metric", color_discrete_map=color_map)
    st.plotly_chart(fig)

    st.write("Created by Daniel Rubio")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download results as CSV", data=csv, file_name="sentiment_analysis_results.csv", mime="text/csv")
