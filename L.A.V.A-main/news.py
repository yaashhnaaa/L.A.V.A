import requests
import json
import pyttsx3
import speech_recognition as sr

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)
rate = engine.setProperty("rate", 170)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def take_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=None)  # Continuously listen

    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio).lower()
        print(f"You: {command}")
        return command
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
        speak("I couldn't understand that. Please try again.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        speak("There was an error. Please try again.")
        return None

def latestnews():
    api_dict = {
        "business": "https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey=0cbc6ba6346543b4ac9e40ab5e4097db",
        "entertainment": "https://newsapi.org/v2/top-headlines?country=in&category=entertainment&apiKey=0cbc6ba6346543b4ac9e40ab5e4097db",
        "health": "https://newsapi.org/v2/top-headlines?country=in&category=health&apiKey=0cbc6ba6346543b4ac9e40ab5e4097db",
        "science": "https://newsapi.org/v2/top-headlines?country=in&category=science&apiKey=0cbc6ba6346543b4ac9e40ab5e4097db",
        "sports": "https://newsapi.org/v2/top-headlines?country=in&category=sports&apiKey=0cbc6ba6346543b4ac9e40ab5e4097db",
        "technology": "https://newsapi.org/v2/top-headlines?country=in&category=technology&apiKey=0cbc6ba6346543b4ac9e40ab5e4097db"
    }

    content = None
    url = None
    while True:
        speak("Which field news do you want, [business], [health], [technology], [sports], [entertainment], [science]")
        field = take_voice_input()

        for key, value in api_dict.items():
            if key.lower() in field.lower():
                url = value
                print(url)
                print("url was found")
                break
            else:
                url = True

        if url is True:
            print("url not found")
            continue

        news = requests.get(url).text
        news = json.loads(news)
        speak("Here is the first news.")

        arts = news["articles"]
        for articles in arts:
            article = articles["title"]
            print(article)
            speak(article)
            news_url = articles["url"]
            print(f"For more info visit: {news_url}")

            while True:
                speak("Say [continue] to continue or [stop] to stop.")
                response = take_voice_input()

                if response and "continue" in response:
                    break
                elif response and "stop" in response:
                    speak("Stopping news updates.")
                    return
                else:
                    speak("Invalid response. Please say [continue] or [stop].")

        speak("That's all")

# Call the function to execute the code
latestnews()
