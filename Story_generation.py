from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from config import API_KEY
from TExttoSpeech import generate_audio
from func import predict_emotion


app = Flask(__name__,static_folder='static', static_url_path='/static')
client = OpenAI(api_key=API_KEY)
messages = [{"role": "system", "content": "You are an intelligent assistant."}]
chat = client.chat.completions.create( model="gpt-3.5-turbo", messages=messages)



is_firstCall=True


@app.route('/')
def chat_ui():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data['user_message']
    bot_response = get_assistant_response(user_message)
    return jsonify({'bot_response': bot_response})

def get_assistant_response(user_message):
    global is_firstCall
    case=0
    words=100
    tone=predict_emotion(user_message)
    print(tone)
    print(user_message)
    while True:
      if is_firstCall:
         text = f"Generate a story with a tone of {tone} with {words} words"
         is_firstCall=False
      elif user_message=='exit':
        text= f"Continue the same story for {words} words and end the story"
        case=1
      else:
        text=f'Continue the same story with the tone of {tone} for another {words} words'
      message = text
      print(is_firstCall)
      if message:
        messages.append({"role": "user", "content": message},)
        chat = client.chat.completions.create( model="gpt-3.5-turbo", messages=messages)
        reply = chat.choices[0].message.content
        if case==1:
          story_segment=f""+str({reply})+""
        else:
          story_segment=f" " + str({reply}) + ".how do you feel now?"
        

        generate_audio(story_segment,"./static/output.mp3",1)
        
        
        messages.append({"role": "assistant", "content": reply})
        
      return reply
    
    


if __name__ == '__main__':
    app.run(debug=True)



