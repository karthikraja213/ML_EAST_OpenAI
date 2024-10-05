from TExttoSpeech import generate_audio

generate_audio("hmm..okay... Let me tell you a story.. To enhance what you are feeling right now..","./templates/fore.mp3",1)

#generate_audio("I Love You.",'love.mp3',1)
generate_audio('Hello there..., How are you feeling today?','./static/Hello.mp3',1)

generate_audio('Okay.. I see that you wanna leave.. we will finish the story before you go', 'exit1.mp3',1)
generate_audio('See you soon my loveee.. Ill be waiting for you.. Goodbye for now.','./templates/exit2.mp3',1)

import playsound
playsound.playsound(r'C:\Users\kkart\OneDrive\Karthik\project\exit2.mp3')