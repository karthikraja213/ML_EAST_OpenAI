import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import sys
import warnings
import neattext.functions as nfx

import joblib

# Load your dataset
df = pd.read_csv('DATA.csv')


df = df[['SIT', 'Field1']]
df = df.rename(columns={'SIT': 'text', 'Field1': 'emotions'})
df = df.dropna(subset=['text'])
text_to_remove = ['Does not apply.', "Doesn't apply.", 'None','[ I have never felt this emotion.]', '[ Never experienced.]','[ Never.]', '[ Not applicable.]', '[ Do not remember any incident.]', "[ No response.]", "NO RESPONSE."]

df = df[~df['text'].isin(text_to_remove)]
df['clean_text'] = df['text'].apply(nfx.remove_userhandles)
df['clean_text'] = df['clean_text'].apply(nfx.remove_stopwords)
df['clean_text'] = df['clean_text'].apply(nfx.remove_punctuations)
df['clean_text'] = df['clean_text'].apply(nfx.remove_hashtags)
df['clean_text'] = df['clean_text'].apply(nfx.remove_urls)
df['clean_text'] = df['clean_text'].apply(nfx.remove_numbers)
df['clean_text'] = df['clean_text'].apply(lambda x: x.replace('á\n', ''))
df['clean_text'] = df['clean_text'].apply(lambda x: x.replace('á\n', ''))
df['clean_text'] = df['clean_text'].apply(lambda x: x.replace('á', ''))
df['clean_text'] = df['clean_text'].apply(lambda x: x.replace('quot', ''))
df['clean_text'] = df['clean_text'].apply(lambda x: x.replace('[ ', ''))
df['clean_text'] = df['clean_text'].apply(lambda x: x.replace(']', ''))
df.drop(df[df['clean_text'] == ''].index, inplace=True)
df.reset_index(drop=True, inplace=True)

# Preprocess the text data
corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['clean_text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Create a Bag of Words model
cv = CountVectorizer(max_features=1500)
X_text = cv.fit_transform(corpus).toarray()

# Select the target variable
y = df['emotions']

# Create a DataFrame with text features and the target variable
X_text_df = pd.DataFrame(X_text, columns=cv.get_feature_names_out())
data_with_text = pd.concat([X_text_df, y], axis=1)

# Convert all feature column names to strings
data_with_text.columns = data_with_text.columns.astype(str)

# Split the data into training and testing sets (you can use your original code)
X_train, X_test, y_train, y_test = train_test_split(data_with_text.drop(columns=['emotions']), data_with_text['emotions'], test_size=0.2, random_state=0)

# Train your SVM model (you can use your original code)
classifierSVM = SVC(kernel='rbf', random_state=0)
classifierSVM.fit(X_train, y_train)

# Test your model on the testing data (you can use your original code)
y_pred2 = classifierSVM.predict(X_test)
accuracy = accuracy_score(y_test, y_pred2)
print("Accuracy:", accuracy)
joblib.dump(classifierSVM, 'SVMmodel.joblib')
joblib.dump(cv, 'count_vectorizer.joblib')




