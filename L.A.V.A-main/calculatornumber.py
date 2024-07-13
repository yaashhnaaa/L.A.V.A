import wolframalpha
import pyttsx3
import speech_recognition as sr

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def WolfRamAlpha(query):
    apikey = "2K958E-EL39X4J874"
    requester = wolframalpha.Client(apikey)
    requested = requester.query(query)

    try:
        answer = next(requested.results).text
        return answer
    except StopIteration:
        speak("The value is not answerable")

def Calc(query):
    if not query:
        speak("Command not recognized. Please try again.")
        return

    term = query.replace("jarvis", "").replace("multiply", "*").replace("plus", "+").replace("minus", "-").replace("divide", "/")

    try:
        result = WolfRamAlpha(term)
        print(result)
        speak(result)
    except:
        speak("The value is not answerable")

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command...")
        speak("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=None)  # Continuously listen

    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print(f"You: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

if __name__ == "__main__":
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[0].id)
    rate = engine.setProperty("rate", 170)

    speak("Hello! How can I assist you today?")

    while True:
        command = recognize_speech()
        if command:
            if "stop" in command:
                speak("Goodbye!")
                break
            else:
                Calc(command)
