import pyttsx3
import speech_recognition as sr
import wolframalpha

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

    def answer_question(self):
        self.talk('I can answer questions on current affairs')
        question = self.take_command()
        self.talk(question)
        app_id = "R2K75H-7ELALHR35X"
        client = wolframalpha.Client(app_id)
        res = client.query(question)
        answer = next(res.results).text
        self.talk(answer)
        print(answer)

    def start(self):
        self.talk("Initializing Lava...")
        while True:
            query = self.take_command().lower()
            if 'exit' in query:
                self.talk("Exiting Lava. Have a good day!")
                break
            elif 'tell me' in query:
                self.answer_question()

# Example usage
jarvis = Jarvis()
jarvis.start()
