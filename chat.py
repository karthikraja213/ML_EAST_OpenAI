
from TExttoSpeech import generate_audio
import playsound
import sys
import openai
from openai import OpenAI
from config import API_KEY
case=0
from flask import Flask, request, jsonify,render_template
from openai import OpenAI
from config import API_KEY  # Ensure you have a config file with your OpenAI API key

app = Flask(__name__)
openai_client = OpenAI(api_key=API_KEY)




playsound.playsound(r'C:\Users\kkart\OneDrive\Karthik\project\Hello.mp3')

count=100
predicted_emotion='joy'
text_input= input("User: ")
#predicted_emotion = predict_emotion(text_input, classifierSVM, cv)
#print(predicted_emotion)
text = f"Generate a story with a tone of {predicted_emotion} with {count} words and prompt a question asking how they feel now?"
# Hello=['Hello Love, How are you feeling today?','Let me tell you a story now to upheave what you are feeling right now']
playsound.playsound(r'C:\Users\kkart\OneDrive\Karthik\project\fore.mp3')

client = OpenAI(api_key=API_KEY)
messages = [ {"role": "system", "content":
			"You are a intelligent assistant."} ]


case=0

try:
  while True:
    message = text
    if message:
      messages.append({"role": "user", "content": message},)
      print(message)
      chat = client.chat.completions.create( model="gpt-3.5-turbo", messages=messages)
      reply = chat.choices[0].message.content
      print(f"E.A.S.T.: {reply}")
      if case==1:
        story_segment=f""+str({reply})+""
      else:
        story_segment=f" " + str({reply}) + ".how do you feel now?"
      generate_audio(story_segment,"output.mp3",1)
      playsound.playsound(r'C:\Users\kkart\OneDrive\Karthik\project\output.mp3')
      messages.append({"role": "assistant", "content": reply})
      if text_input=='exit':
        playsound.playsound(r'C:\Users\kkart\OneDrive\Karthik\project\exit2.mp3')
        sys.exit("Exiting")
        
      text_input = input ("How do you feel now? :")

      if text_input=='exit':
        playsound.playsound(r'C:\Users\kkart\OneDrive\Karthik\project\exit1.mp3')
        text2=f"Continue the same story with the same emotion for {count} words and end the story"
        case=1
        
      else:
        #predicted_emotion = predict_emotion(text_input, classifierSVM, cv)
        print(predicted_emotion)
        text2 = f"Coninue the same story with {predicted_emotion} for another {count} words"
      text=text2


except SystemExit:
      # Handle the SystemExit exception gracefully
      print("THE END")

