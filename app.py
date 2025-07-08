
import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: white;
        margin-top: 0.5rem;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load NLTK resources
@st.cache_resource
def load_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))
    except:
        return set()

stop_words = load_nltk_data()
stemmer = PorterStemmer()

# Load models
@st.cache_resource
def load_models():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'vectorizer.pkl' and 'sentiment_model.pkl' are in the same directory.")
        return None, None

vectorizer, model = load_models()

# Configuration
label_map = {0: "Negative", 1: "Positive"}
emoji_map = {"Negative": "😠", "Positive": "😊"}
color_map = {"Negative": "#ff6b6b", "Positive": "#4ecdc4"}

# Preprocessing function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

def get_confidence(prediction_proba):
    if hasattr(prediction_proba, 'max'):
        return prediction_proba.max()
    return 0.75

def main():
    st.markdown("""
    <div class="main-header">
        <h1>💬 Twitter Sentiment Analyzer</h1>
        <p>Analyze the sentiment of tweets with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Settings")
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_wordcloud = st.checkbox("Generate Word Cloud", value=True)
        show_charts = st.checkbox("Show Charts", value=True)
        wc_width, wc_height, wc_max_words = 800, 400, 100
        if show_charts:
            st.subheader("Chart Options")
            chart_type = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart", "Donut Chart"])

    st.subheader("📝 Input Tweets")

    uploaded_file = st.file_uploader("Upload CSV file containing tweets", type=["csv"])
    tweets_from_csv = []

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if 'text' in df_upload.columns:
                tweets_from_csv = df_upload['text'].astype(str).tolist()
                st.success(f"Loaded {len(tweets_from_csv)} tweets from CSV.")
            else:
                st.error("CSV must have a 'text' column.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    default_text = st.session_state.get('sample_input', '\n'.join(tweets_from_csv) if tweets_from_csv else '')
    user_input = st.text_area(
        "Enter tweets (one per line):",
        value=default_text,
        height=200,
        placeholder="Paste tweets here...\nExample:\nI love this product!\nThis is terrible service."
    )

    if vectorizer is None or model is None:
        return

    if st.button("🚀 Analyze Sentiments", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text or upload a file.")
        else:
            with st.spinner("Analyzing sentiments..."):
                tweets = [tweet.strip() for tweet in user_input.strip().split('\n') if tweet.strip()]
                df = pd.DataFrame(tweets, columns=['text'])
                df['clean_text'] = df['text'].apply(clean_text)
                vect = vectorizer.transform(df['clean_text'])
                try:
                    prediction_proba = model.predict_proba(vect)
                    df['confidence'] = [get_confidence(p) for p in prediction_proba]
                except:
                    df['confidence'] = [0.75] * len(tweets)
                predictions = model.predict(vect)
                df['sentiment'] = [label_map[p] for p in predictions]
                df['emoji'] = df['sentiment'].map(emoji_map)
                df['color'] = df['sentiment'].map(color_map)

                st.header("📋 Analysis Results")
                sentiment_counts = df['sentiment'].value_counts()
                total_tweets = len(df)
                col1, col2, col3 = st.columns(3)

                with col1:
                    positive_count = sentiment_counts.get('Positive', 0)
                    st.metric("😊 Positive", f"{positive_count}", f"{(positive_count/total_tweets)*100:.1f}%")

                with col2:
                    negative_count = sentiment_counts.get('Negative', 0)
                    st.metric("😠 Negative", f"{negative_count}", f"{(negative_count/total_tweets)*100:.1f}%")

                with col3:
                    st.metric("🎯 Avg Confidence", f"{df['confidence'].mean():.2f}", f"{df['confidence'].mean()*100:.1f}%")

                st.subheader("📄 Detailed Results")
                display_df = df.copy()
                if show_confidence:
                    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(display_df[['text', 'sentiment', 'emoji', 'confidence']], use_container_width=True)
                else:
                    st.dataframe(display_df[['text', 'sentiment', 'emoji']], use_container_width=True)

                if show_charts:
                    st.subheader("📊 Visualization")
                    if chart_type == "Bar Chart":
                        fig = px.bar(
                            x=sentiment_counts.index,
                            y=sentiment_counts.values,
                            color=sentiment_counts.index,
                            color_discrete_map=color_map,
                            title="Sentiment Distribution"
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Pie Chart":
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            color=sentiment_counts.index,
                            color_discrete_map=color_map,
                            title="Sentiment Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "Donut Chart":
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            color=sentiment_counts.index,
                            color_discrete_map=color_map,
                            title="Sentiment Distribution",
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)

                if show_wordcloud:
                    st.subheader("☁️ Word Cloud")
                    all_words = " ".join(df['clean_text'].dropna().tolist())
                    if all_words.strip():
                        wc = WordCloud(
                            width=wc_width,
                            height=wc_height,
                            background_color='white',
                            max_words=wc_max_words,
                            colormap='viridis'
                        ).generate(all_words)
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis('off')
                        st.pyplot(fig_wc)
                    else:
                        st.info("No words to display")

                st.subheader("📎 Export Results")
                col1, col2 = st.columns(2)

                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📅 Download as CSV",
                        data=csv,
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    summary = f"""
                    Sentiment Analysis Report
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                    Total Tweets Analyzed: {total_tweets}
                    Positive Tweets: {positive_count} ({(positive_count/total_tweets)*100:.1f}%)
                    Negative Tweets: {negative_count} ({(negative_count/total_tweets)*100:.1f}%)
                    Average Confidence: {df['confidence'].mean():.2f}
                    """
                    st.download_button(
                        label="📄 Download Summary",
                        data=summary,
                        file_name=f"sentiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()



