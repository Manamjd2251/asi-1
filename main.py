import wikipedia
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

geographic_titles = [
    "Madrid", "Brazil", "Berlin", "United States", "Rome", "Pacific Ocean"
]

non_geographic_titles = [
    "Mathematics", "Computer science", "Economics", "Literature", "Philosophy", "Banana"
]

def fetch_wikipedia_texts(titles):
    texts = []
    for title in titles:
        try:
            page = wikipedia.page(title)
            texts.append(page.content)
        except Exception as e:
            print(f"Failed to fetch '{title}': {e}")
    return texts

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [ps.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

geo_texts = fetch_wikipedia_texts(geographic_titles)
non_geo_texts = fetch_wikipedia_texts(non_geographic_titles)

texts = geo_texts + non_geo_texts
labels = ["geographic"] * len(geo_texts) + ["non-geographic"] * len(non_geo_texts)

print("Preprocessing texts...")
texts_processed = [preprocess_text(t) for t in texts]

print("Vectorizing...")
vectorizer = CountVectorizer(max_features=3000)
X = vectorizer.fit_transform(texts_processed)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42, stratify=labels
)

print("Training Naive Bayes classifier...")
nb = MultinomialNB()
nb.fit(X_train, y_train)

print("Predicting...")
y_pred = nb.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

"result : Confusion Matrix:
[[2 0]
 [1 1]]

Classification Report:
                precision    recall  f1-score   support

    geographic       0.67      1.00      0.80         2
non-geographic       1.00      0.50      0.67         2

      accuracy                           0.75         4
     macro avg       0.83      0.75      0.73         4
  weighted avg       0.83      0.75      0.73         4"

"Wikipedia Text Classification – Geographic vs. Non-Geographic

This project implements a Naive Bayes classifier using Python and NLTK to classify Wikipedia pages as either geographic or non-geographic.

🧠 Pipeline Overview

Wikipedia Text Collection:
Pages like "Madrid" and "Brazil" were used as geographic examples.
Pages like "Mathematics" and "Philosophy" were used as non-geographic examples.
Text Preprocessing:
Tokenized using nltk.word_tokenize
Lowercased, stopwords removed (nltk.corpus.stopwords)
Words stemmed using PorterStemmer
Feature Extraction:
Used CountVectorizer (BoW, max 3000 features)
Model Training:
Trained MultinomialNB using scikit-learn
Stratified train-test split (70% / 30%)
Evaluation:
Confusion matrix and classification report printed
Handled zero_division safely"
