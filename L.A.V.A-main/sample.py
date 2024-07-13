import sys
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, QTime, QDate, Qt, QThread
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QApplication, QMainWindow
from JarvisUi import Ui_JarvisUI  # Assuming you have a separate UI file

import pyttsx3
import speech_recognition as sr
import pywhatkit
import datetime
import wikipedia
import webbrowser
from pytube import YouTube
import requests  # Add this import for handling API requests

NEWS_API_KEY = '0cbc6ba6346543b4ac9e40ab5e4097db'  
OPEN_CAGE_API_KEY = 'adfb7e4ee237405fb829e91a47165bfc' 

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
        self.talk("Sorry I didn't get that. Can you please say it again?")
    
    def get_daily_news(self):
        self.talk("Fetching today's top headlines...")
        news_url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}'
        try:
            response = requests.get(news_url)
            news_data = response.json()

            if news_data['status'] == 'ok':
                articles = news_data['articles']
                for article in articles[:5]:  # Speak the top 5 headlines
                    title = article['title']
                    self.talk(title)
            else:
                self.talk("Sorry, I couldn't fetch the news at the moment.")

        except Exception as e:
            print(e)
            self.talk("Sorry, I encountered an error while fetching the news.")
    
    def track_location(self):
        self.talk("Sure, I can help you track your location. Please wait a moment.")
        
        # Use a location tracking API, in this case, OpenCage Geocoding API
        try:
            ip_info = requests.get('https://ipinfo.io').json()
            ip_location = ip_info.get('loc', '').split(',')
            latitude, longitude = ip_location
            
            # Use OpenCage Geocoding API to get the detailed location information
            location_url = f'https://api.opencagedata.com/geocode/v1/json?q={latitude}+{longitude}&key={OPEN_CAGE_API_KEY}'
            location_info = requests.get(location_url).json()
            
            # Extract relevant information from the API response
            city = location_info['results'][0]['components']['city']
            country = location_info['results'][0]['components']['country']

            self.talk(f"Your current location is in {city}, {country}.")
        except Exception as e:
            print(e)
            self.talk("Sorry, I couldn't retrieve your location at the moment.")

    def yt_google_commands(self, command):
        print(command)
        if 'play' in command:
            self.talk("Boss can you please say the name of the song")
            song = self.take_Command()
            if "play" in song:
                song = song.replace("play", "")
            self.talk('playing ' + song)
            print(f'playing {song}')
            pywhatkit.playonyt(song)
            print('playing')
        elif "download" in command:
            self.talk("Boss please enter the YouTube video link which you want to download")
            link = input("Enter the YOUTUBE video link: ")
            yt = YouTube(link)
            yt.streams.get_highest_resolution().download()
            self.talk(f"Boss downloaded {yt.title} from the link you given into the main folder")
        elif 'youtube' in command:
            self.talk('opening your youtube')
            webbrowser.open('https://www.youtube.com/')
        elif 'search' in command or 'google' in command:
            self.talk("Boss, what should I search on Google..")
            search_query = self.take_Command()
            webbrowser.open(f"https://www.google.com/search?q={search_query}")
        elif 'wikipedia' in command:
            self.talk("Boss, please tell me what you want to know on Wikipedia")
            wiki_query = self.take_Command()
            result = wikipedia.summary(wiki_query, sentences=2)
            print(result)
            self.talk(result)
        elif 'netflix' in command:
            self.talk("Boss, opening Netflix for you")
            webbrowser.open('https://www.netflix.com/')
        elif 'prime' in command:
            self.talk('Boss opening Amazon prime video for you')
            webbrowser.open('https://www.primevideo.com/offers/nonprimehomepage/ref=dv_web_force_root')
        elif 'facebook' in command:
            self.talk("Boss, opening Facebook for you")
            webbrowser.open('https://www.facebook.com/')
        elif 'instagram' in command:
            self.talk("Boss, opening Instagram for you")
            webbrowser.open('https://www.instagram.com/')
        elif 'gmail' in command:
            self.talk("Boss, opening Gmail for you")
            webbrowser.open('https://mail.google.com/')
        else:
            self.No_result_found()

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

    def internet_speed(self):
        self.talk("Checking internet speed...")
        # Perform internet speed check here (you can add your internet speed code here)
        self.talk("Internet speed check complete.")

    def Fun(self, command):
        print(command)
        if 'your name' in command:
            self.talk("My name is Lava")
        elif 'news' in command:
            self.get_daily_news()
        elif 'location' in command:
            self.track_location()


        elif 'who are you' in command:
            self.talk("I am Lava, your virtual assistant.")
        # Add more conditions as needed
        elif 'google' in command or 'search' in command:
            self.talk("Boss, what should I search on Google..")
            search_query = self.take_Command()
            webbrowser.open(f"https://www.google.com/search?q={search_query}")
        else:
            self.yt_google_commands(command)

# Creating a class to manage the main thread
class MainThread(QThread):
    def __init__(self):
        super(MainThread, self).__init__()

    def run(self):
        start_execution = Jarvis()
        start_execution.start()

# Your existing code for the UI
class Main(QMainWindow):
    cpath = ""
    global startExecution  # Declare startExecution as global

    def __init__(self, path):
        self.cpath = path
        super().__init__()
        self.ui = Ui_JarvisUI(path=current_path)
        self.ui.setupUi(self)
        self.ui.pushButton_4.clicked.connect(self.startTask)
        self.ui.pushButton_3.clicked.connect(self.close)

    def startTask(self):
        global startExecution  # Declare startExecution as global
        startExecution = MainThread()  # Re-initialize startExecution
        self.ui.movie = QtGui.QMovie(rf"{self.cpath}\UI\ironman1.gif")
        self.ui.label_2.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie(rf"{self.cpath}UI\artifact.gif")
        self.ui.label_3.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie(rf"{self.cpath}\UI\circle.gif")
        self.ui.label_4.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie(rf"{self.cpath}\UI\lines1.gif")
        self.ui.label_7.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie(rf"{self.cpath}\UI\ironman3.gif")
        self.ui.label_8.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie(rf"{self.cpath}\UI\circle.gif")
        self.ui.label_9.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie(rf"{self.cpath}\UI\powersource.gif")
        self.ui.label_12.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie(rf"{self.cpath}\UI\powersource.gif")
        self.ui.label_13.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie(rf"{self.cpath}\UI\ironman3_flipped.gif")
        self.ui.label_16.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie(rf"{self.cpath}\UI\Sujith.gif")
        self.ui.label_17.setMovie(self.ui.movie)
        self.ui.movie.start()
        timer = QTimer(self)
        timer.timeout.connect(self.showTime)
        timer.start(1000)
        startExecution.start()

    def showTime(self):
        current_time = QTime.currentTime()
        current_date = QDate.currentDate()
        label_time = current_time.toString('hh:mm:ss')
        label_date = current_date.toString(Qt.ISODate)
        self.ui.textBrowser.setText(label_date)
        self.ui.textBrowser_2.setText(label_time)

current_path = os.getcwd()
app = QApplication(sys.argv)
jarvis = Main(path=current_path)
jarvis.show()

exit(app.exec_())
