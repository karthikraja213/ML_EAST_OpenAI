import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from datetime import datetime
from Story_generation import get_assistant_response
import pygame
import time
from tkinter import font as tkfont




class ChatBotApp:
    
    def __init__(self, root):
        self.root = root

        self.root.title("MAK Bot")

        # Create widgets
        self.create_widgets()
        pygame.mixer.init()
        self.play_sound('./static/Hello.mp3')
        self.count=True
        

    

    def create_widgets(self):
        tfont = ("Arial", 14)
        # Chat history display
        self.chat_history = scrolledtext.ScrolledText(self.root, width=50, height=20, state='disabled',font=tfont, bg="black", fg="white")
        self.chat_history.grid(row=0, column=0, padx=10, pady=10, columnspan=3)
       
        # User input entry
        self.user_input = tk.Entry(self.root, width=40,font=tfont)
        self.user_input.grid(row=1, column=0, padx=10, pady=10)

        # Send button
        send_button = tk.Button(self.root, text="Send", command=self.send_message,bg="blue", fg="white")
        send_button.grid(row=1, column=1, padx=5, pady=10)

        # Clear button
        clear_button = tk.Button(self.root, text="Clear", command=self.clear_chat)
        clear_button.grid(row=1, column=2, padx=5, pady=10)

         # Play button
        pygame.mixer.init()
        play_button = tk.Button(self.root, text="Talk to me again", command=self.playsoundreplay)
        play_button.grid(row=2, column=1, padx=5, pady=10, columnspan=4)

    def send_message(self):
        
        if self.count==True:
            pygame.mixer.init()
            print("playing fore")
            self.play_sound('./templates/fore.mp3')
            time.sleep(5)
            self.count=False
        
        user_message = self.user_input.get()
        if user_message:
            self.display_message("You", user_message)
            if user_message=="exit":    
                self.play_sound('./static/exit1.mp3')
                while pygame.mixer.music.get_busy():
                    continue
                bot_response = self.get_bot_response(user_message)
                self.display_message("Assistant:", bot_response)
                pygame.mixer.quit()
                self.play_sound('./static/output.mp3')
                self.root.after(5000, self.play_last_audio)
                
            else:
                
                pygame.mixer.quit()
                bot_response = self.get_bot_response(user_message)
                self.root.after(1000, lambda: self.display_message("Assistant", bot_response))
                self.play_sound('./static/output.mp3')
                


            # Clear the user input field
            self.user_input.delete(0, 'end')
        else:
            messagebox.showinfo("Error", "Please enter a message!")

    def display_message(self, sender, message):
        
        formatted_message = f"{sender}: {message}\n"

        # Enable the text widget to modify its content
        self.chat_history.config(state='normal')
        self.chat_history.insert(tk.END, formatted_message)

        # Disable the text widget to prevent further modifications
        self.chat_history.config(state='disabled')

        # Scroll to the bottom to show the latest message
        self.chat_history.yview(tk.END)

    def clear_chat(self):
        # Clear the chat history
        self.chat_history.config(state='normal')
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state='disabled')
    
    def play_sound(self,path):
        pygame.mixer.init()
        pygame.mixer.music.load(path) 
        pygame.mixer.music.play()
    
    def playsoundreplay(self):
        pygame.mixer.init()
        pygame.mixer.music.load('static/output.mp3') 
        pygame.mixer.music.play()

    def get_bot_response(self, user_message):
        
        return get_assistant_response(user_message)
    
    def play_last_audio(self):
        while pygame.mixer.music.get_busy():
            continue
        self.play_sound('./static/exit2.mp3')
        while pygame.mixer.music.get_busy():
            continue
        time.sleep(2)
        pygame.mixer.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatBotApp(root)
    root.mainloop()
