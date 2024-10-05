
from pathlib import Path
from openai import OpenAI
import requests

def generate_audio(text,outputfile,rate):
  client = OpenAI(api_key='<API KEY>')

  response = client.audio.speech.create(
  model="tts-1",
  voice="nova",
  input=text,
  speed=rate
 )

  response.stream_to_file(outputfile)


  response.close()