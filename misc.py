import joblib

loaded_model = joblib.load('SVMmodel.joblib')

loaded_cv=joblib.load('count_vectorizer.joblib')



text="happy"

def predict_emotion(text_input):
    new_corpus = [text_input]
    new_X_text = loaded_cv.transform(new_corpus).toarray()
    predicted_emotion = loaded_model.predict(new_X_text)[0]
    return predicted_emotion
print(predict_emotion(text))