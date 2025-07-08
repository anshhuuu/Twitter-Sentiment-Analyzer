## Twitter Sentiment Analyzer
A web-based sentiment analysis tool for tweets built using Streamlit, Logistic Regression, and NLTK. Upload tweets (as CSV) or enter them manually to visualize sentiment distribution and generate word clouds.

##  Features
 Classifies tweets as Positive or Negative

 Visualizes sentiment distribution using bar, pie, or donut charts

 Generates a word cloud of frequently used terms

 Supports CSV file upload or manual input

 Exports results as CSV and summary report

##  Model Details
Model: Logistic Regression

Vectorizer: TF-IDF (max_features=10000)

Preprocessing:

URL/user/hashtag removal

Lowercasing

Stopword removal

Stemming with PorterStemmer

Trained on: Twitter dataset with sentiment labels (0 = Negative, 1 = Positive)

## Project Structure
```
Twitter-Sentiment-Analyzer/
│
├── app.py / streamlit_sentiment_app.py 
├── model.ipynb                              
├── vectorizer.pkl                           
├── sentiment_model.pkl                     
├── training.data.csv                        
└── README.md  

```

##  Installation
1. Clone the repo
```
git clone https://github.com/anshhuuu/Twitter-Sentiment-Analyzer.git
cd Twitter-Sentiment-Analyzer

```
2. Install dependencies

```
pip install -r requirements.txt
```
3. Download NLTK data

```
import nltk
nltk.download('stopwords')
```

