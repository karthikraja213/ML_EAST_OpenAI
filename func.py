import joblib
from sklearn.preprocessing import LabelEncoder

loaded_model = joblib.load('SVMmodel1.joblib')
loaded_cv=joblib.load('tfidf_vectorizer.joblib')

loaded_sentiment_encoder = joblib.load('sentiment_encoder.joblib')

def map_emotion_labels(predicted_emotion):
    emotion_mapping = {
        'Positive': 'optimistic',
        'Negative': 'pessimistic',
        'Neutral': 'objective',
        'Irrelevant':'Mixed'
    }

    new_label = emotion_mapping.get(predicted_emotion, 'unknown')

    return new_label


def predict_emotion(text_input):
    # Transform the text input using the pre-fitted TF-IDF Vectorizer
    new_corpus = [text_input]
    new_X_text = loaded_cv.transform(new_corpus).toarray()

    # Make predictions using the model
    predicted_emotion_coded = loaded_model.predict(new_X_text)  
    predicted_emotion = loaded_sentiment_encoder.inverse_transform(predicted_emotion_coded)[0]
    
    new_label = map_emotion_labels(predicted_emotion)

    return new_label

print(predict_emotion("i am not feeling okay now."))
