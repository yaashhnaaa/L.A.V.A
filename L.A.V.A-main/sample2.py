import pyttsx3
import speech_recognition as sr
import pywhatkit
import datetime
import wikipedia
import webbrowser
from pytube import YouTube
import requests
import speedtest as st  # Change the import statement

class Jarvis:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[1].id)
        self.r = sr.Recognizer()

    def talk(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def greet(self):
        hour = int(datetime.datetime.now().hour)
        if 0 <= hour < 12:
            self.talk("Good Morning Boss!")
        elif 12 <= hour < 18:
            self.talk("Good Afternoon Boss!")
        else:
            self.talk("Good Evening Boss!")
        self.talk("How can I assist you today?")

    def take_Command(self):
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
            print("Say that again please...")
            return "None"
        return query.lower()

    def No_result_found(self):
        self.talk("Sorry, I didn't get that. Can you please say it again?")

    def yt_google_commands(self, command):
        if 'play' in command:
            self.play_music(command)
        elif 'download' in command:
            self.download_video()
        elif 'youtube' in command:
            self.open_youtube()
        elif 'search' in command or 'google' in command:
            self.search_google()
        elif 'wikipedia' in command:
            self.search_wikipedia()
        elif 'netflix' in command:
            self.open_netflix()
        elif 'prime' in command:
            self.open_prime_video()
        elif 'facebook' in command:
            self.open_facebook()
        elif 'instagram' in command:
            self.open_instagram()
        elif 'gmail' in command:
            self.open_gmail()
        elif 'internet speed' in command:
            self.check_internet_speed()
        else:
            self.No_result_found()

    def play_music(self, command):
        self.talk("Boss, can you please say the name of the song?")
        song = self.take_Command()
        if "play" in song:
            song = song.replace("play", "")
        self.talk(f'Playing {song}')
        print(f'Playing {song}')
        pywhatkit.playonyt(song)

    def download_video(self):
        self.talk("Boss, please enter the YouTube video link you want to download.")
        link = input("Enter the YouTube video link: ")
        yt = YouTube(link)
        yt.streams.get_highest_resolution().download()
        self.talk(f"Boss, downloaded {yt.title} from the link you provided.")

    def open_youtube(self):
        self.talk('Opening YouTube for you.')
        webbrowser.open('https://www.youtube.com/')

    def search_google(self):
        self.talk("Boss, what should I search on Google?")
        search_query = self.take_Command()
        webbrowser.open(f"https://www.google.com/search?q={search_query}")

    def search_wikipedia(self):
        self.talk("Boss, please tell me what you want to know on Wikipedia.")
        wiki_query = self.take_Command()
        result = wikipedia.summary(wiki_query, sentences=2)
        print(result)
        self.talk(result)

    def open_netflix(self):
        self.talk("Boss, opening Netflix for you.")
        webbrowser.open('https://www.netflix.com/')

    def open_prime_video(self):
        self.talk('Boss, opening Amazon Prime Video for you.')
        webbrowser.open('https://www.primevideo.com/offers/nonprimehomepage/ref=dv_web_force_root')

    def open_facebook(self):
        self.talk("Boss, opening Facebook for you.")
        webbrowser.open('https://www.facebook.com/')

    def open_instagram(self):
        self.talk("Boss, opening Instagram for you.")
        webbrowser.open('https://www.instagram.com/')

    def open_gmail(self):
        self.talk("Boss, opening Gmail for you.")
        webbrowser.open('https://mail.google.com/')

    def check_internet_speed(self):
        self.talk("Checking internet speed...")
        wifi = st.Speedtest()  # Change the object creation
        upload_net = wifi.upload() / 1024 / 1024  # Megabyte = 1024 Bytes
        download_net = wifi.download() / 1024 / 1024
        print("Wifi Upload Speed is", upload_net)
        print("Wifi download speed is ", download_net)
        self.talk(f"Wifi download speed is {download_net:.2f} megabytes per second")
        self.talk(f"Wifi Upload speed is {upload_net:.2f} megabytes per second")

    def start(self):
        self.talk("Initializing Lava...")
        self.greet()
        while True:
            query = self.take_Command().lower()
            if 'exit' in query:
                self.talk("Exiting Lava. Have a good day!")
                break
            elif 'start' in query:
                self.talk("Lava is now active. How can I assist you?")
                self.internet_speed()
            else:
                self.Fun(query)
                
    def Fun(self, command):
        if 'your name' in command:
            self.talk("My name is Lava.")
        elif 'who are you' in command:
            self.talk("I am Lava, your virtual assistant.")
        else:
            self.yt_google_commands(command)

# Run Jarvis
jarvis_instance = Jarvis()
jarvis_instance.start()
