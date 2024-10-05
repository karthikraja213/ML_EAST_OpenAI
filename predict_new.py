
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import joblib


# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df_sentiment = pd.read_csv('twitter_training.csv')

df_sentiment.columns = ['TweetID', 'entity', 'Sentiment', 'TweetContent']
df_sentiment = df_sentiment.dropna()
missing_values = df_sentiment.isna().sum()

df_sentiment = df_sentiment[~df_sentiment['Sentiment'].isin(['Irrelevant'])] 

def tokenize(text):
    return word_tokenize(text)

def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def remove_non_alphabetic(tokens):
    return [word for word in tokens if word.isalpha()]

def stem_words(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens)
    tokens = remove_non_alphabetic(tokens)
    tokens = stem_words(tokens)
    return tokens

# Apply preprocessing to the 'Text' column
df_sentiment['Processed_Text'] = df_sentiment['TweetContent'].apply(preprocess_text)

# Initialize another label encoder for Sentiment
sentiment_encoder = LabelEncoder()

# Fit and transform the Sentiment column
df_sentiment.loc[:, 'Sentiment_Encoded'] = sentiment_encoder.fit_transform(df_sentiment['Sentiment'])

# Get the mapping of sentiment labels to encoded numbers
label_mapping = dict(zip(sentiment_encoder.classes_, sentiment_encoder.transform(sentiment_encoder.classes_)))

print("Sentiment to Number Mapping:")
print(label_mapping)

from sklearn.feature_extraction.text import TfidfVectorizer

# Join the list of words in Processed_Text into a single string per row
df_sentiment['Processed_Text_Joined'] = df_sentiment['Processed_Text'].apply(lambda x: ' '.join(x))

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the Processed_Text_Joined column
tfidf_matrix = tfidf_vectorizer.fit_transform(df_sentiment['Processed_Text_Joined'])

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Combine the TF-IDF features with the encoded GameName and Sentiment
#X = hstack((tfidf_matrix, df_sentiment[['GameName_Encoded']].values))
X = tfidf_matrix
y = df_sentiment['Sentiment_Encoded']  # assuming you want to predict Sentiment

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize classifiers
#nb_classifier = MultinomialNB()
svm_classifier = SVC()
#rf_classifier = RandomForestClassifier()

# Train the classifiers
#nb_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
#rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
#nb_predictions = nb_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)
#predictions = rf_classifier.predict(X_test)

# Evaluate accuracy
#nb_accuracy = accuracy_score(y_test, nb_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
#rf_accuracy = accuracy_score(y_test, predictions)

# Print accuracy for each classifier
#print(f'Multinomial Naive Bayes Accuracy: {nb_accuracy:.4f}')
print(f'Support Vector Machine Accuracy: {svm_accuracy:.4f}')
#print(f'Random Forest Accuracy: {rf_accuracy:.4f}')

import numpy as np  # Ensure you import NumPy

# Example usage of the prediction function
def predict_emotion(text_input, svm_model, tfidf_vectorizer):
    # Transform the text input using the pre-fitted TF-IDF Vectorizer
    new_corpus = [text_input]
    new_X_text = tfidf_vectorizer.transform(new_corpus).toarray()

    # Make predictions using the model
    predicted_emotion = svm_model.predict(new_X_text)  # Remove reshape(1, -1)

    return predicted_emotion

# Example text input
text_input = "Happy"  # Replace with your own text input

# Use the same TF-IDF Vectorizer used to train the model
tfidf_vectorizer_for_prediction = tfidf_vectorizer

# Ensure the vectorizer has the same vocabulary as the one used during training
# (No need to fit again, just transform based on the existing vocabulary)
new_X_text = tfidf_vectorizer_for_prediction.transform([text_input]).toarray()

# Make sure to use the correct SVM model (svm_classifier) you trained
# Replace `svm_classifier` with the actual SVM model you trained
predicted_emotion = predict_emotion(text_input, svm_classifier, tfidf_vectorizer_for_prediction)
print("Predicted Emotion:", predicted_emotion)

joblib.dump(svm_classifier, 'SVMmodel1.joblib')
joblib.dump(tfidf_vectorizer_for_prediction, 'tfidf_vectorizer.joblib')

joblib.dump(sentiment_encoder, 'sentiment_encoder.joblib')

