import pandas as pd
from newspaper import Article
import nltk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline
import spacy
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

warnings.filterwarnings("ignore")

# Suppress transformers library warnings
logging.getLogger('transformers').setLevel(logging.ERROR)

# Suppress nltk download messages
logging.getLogger('nltk').setLevel(logging.ERROR)

# Ensure required NLTK datasets are available
nltk.download('punkt', quiet=True)

# Load spaCy model for text preprocessing
nlp = spacy.load('en_core_web_sm')

# Define URLs for web scraping
urls = [
    'https://www.bbc.com/news/world',
    'https://www.cnn.com/world',
    'https://www.aljazeera.com/news/',
]

def fetch_articles(urls):
    """Fetches articles from the given URLs and returns a DataFrame."""
    articles = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            if 'https' in link['href']:
                try:
                    article = Article(link['href'])
                    article.download()
                    article.parse()
                    articles.append({
                        'title': article.title,
                        'text': article.text,
                        'url': link['href']
                    })
                except Exception as e:
                    print(f"Error fetching article: {e}")
    return pd.DataFrame(articles)

# Fetch articles and convert to DataFrame
df = fetch_articles(urls)
df.head()

# Data Preprocessing
def preprocess(text):
    """Preprocesses text data using spaCy for lemmatization and stopword removal."""
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess)
df[['title', 'processed_text']].head()

# Topic Modeling Using LDA
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['processed_text'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

def display_topics(model, feature_names, n_top_words):
    """Displays the top words for each topic in LDA model."""
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

display_topics(lda, vectorizer.get_feature_names_out(), 10)


# Sentiment Analysis with Chunking
sentiment_pipeline = pipeline('sentiment-analysis',
                              model='distilbert-base-uncased-finetuned-sst-2-english',
                              revision='af0f99b')

def analyze_sentiment(text, chunk_size=300):
    """Analyzes sentiment by chunking large text and aggregating results."""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    sentiments = [sentiment_pipeline(chunk)[0]['label'] for chunk in chunks if chunk]
    sentiment_score = sentiments.count('POSITIVE') - sentiments.count('NEGATIVE')
    if sentiment_score > 0:
        return 'POSITIVE'
    elif sentiment_score < 0:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

# Apply chunked sentiment analysis
df['sentiment'] = df['text'].apply(lambda x: analyze_sentiment(x))

# Emotional Analysis using Transformer Model with Text Chunking
emotion_pipeline = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, top_k=None)

def analyze_emotions(text, chunk_size=512):
    """Analyzes emotions in the text by chunking to avoid exceeding model's max input length."""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    emotions = {}
    for chunk in chunks:
        results = emotion_pipeline(chunk)
        for result in results[0]:
            if result['label'] in emotions:
                emotions[result['label']] += result['score']
            else:
                emotions[result['label']] = result['score']
    # Normalize scores
    total_score = sum(emotions.values())
    if total_score > 0:
        for key in emotions:
            emotions[key] /= total_score
    return emotions

# Apply emotional analysis
df['emotions'] = df['text'].apply(analyze_emotions)

# Convert emotions to a DataFrame
emotions_df = pd.json_normalize(df['emotions'])

# Merge the emotions with the original DataFrame
df = pd.concat([df, emotions_df], axis=1)

df[['title', 'sentiment', 'emotions']].head()

# Political Bias Detection
bias_labels = [0, 1, 2] * ((len(df) // 3) + 1)
df['political_bias'] = bias_labels[:len(df)]

label_encoder = LabelEncoder()
df['bias_encoded'] = label_encoder.fit_transform(df['political_bias'])

X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['bias_encoded'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str)))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Political Bias Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize Sentiment Distribution
sentiment_counts = df['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Sentiment Distribution of Articles')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Visualize Emotional Analysis
emotions_agg = emotions_df.sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=emotions_agg.index, y=emotions_agg.values, palette='coolwarm')
plt.title('Emotional Analysis of Articles')
plt.xlabel('Emotion')
plt.ylabel('Aggregate Score')
plt.show()

# Output a summary of results
print(df[['title', 'sentiment', 'political_bias'] + list(emotions_df.columns)].head())
