import pyttsx3
import speech_recognition as sr
import pyautogui
import subprocess
import time
import os

class Jarvis:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[1].id)
        self.r = sr.Recognizer()

    def talk(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def take_command(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.r.pause_threshold = 1
            audio = self.r.listen(source)
        try:
            print("Recognizing...")
            query = self.r.recognize_google(audio, language='en-in')
            print(f"User said: {query}")
        except Exception as e:
            print(e)
            print("Say that again, please...")
            return "None"
        return query.lower()

    def open_notepad(self):
        subprocess.Popen(["notepad.exe"])

    def save_notepad(self):
        # Simulate pressing Ctrl + S to save
        pyautogui.hotkey('ctrl', 's')
        time.sleep(1)  # Wait for the save dialog to appear

        # You may add logic here to handle the save dialog if needed
        # For example, type the file name and press Enter

    def close_notepad(self):
        # Simulate pressing Alt + F4 to close Notepad
        pyautogui.hotkey('alt', 'f4')
        time.sleep(1)  # Wait for Notepad to close

        # You may add logic here to handle the confirmation dialog if needed
        # For example, press 'n' for 'No' or 'enter' to confirm

    def notepad_automation(self):
        self.talk("Opening Notepad. What would you like me to write?")
        content_to_write = self.take_command()

        self.open_notepad()
        pyautogui.write(content_to_write)

        while True:
            self.talk("Do you want to add more? Say 'yes' or 'no'")
            response = self.take_command()
            if 'no' in response:
                break
            elif 'yes' in response:
                self.talk("What would you like to add?")
                additional_content = self.take_command()
                pyautogui.write(additional_content)
            else:  
                self.talk("Sorry, I didn't get that.")

        self.talk("Do you want to save changes? Say 'yes' or 'no'")
        save_response = self.take_command()
        if 'yes' in save_response:
            self.save_notepad()
            self.talk("Changes saved.")
        else:
            self.talk("Changes not saved.")

        self.close_notepad()
        self.talk("Notepad automation complete.")

def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Assistant: Please say a command or 'exit' to stop.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Assistant: I couldn't understand your command.")
        return None
    except sr.RequestError as e:
        print(f"Assistant: Unable to request results from Google Speech Recognition service; {e}")
        return None

def main():
    jarvis = Jarvis()
    while True:
        command = recognize_speech()

        if command is None:
            continue

        if command == "exit":
            print("Exiting...")
            break
        elif command:
            jarvis.notepad_automation()

        # Add a delay between commands
        time.sleep(2)

if __name__ == "__main__":
    main()
