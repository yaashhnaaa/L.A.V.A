import pyttsx3
import speech_recognition as sr

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

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

    reminder_file_path = r"C:\Users\Dell\Desktop\Remember.txt"

    print("Hello! How can I assist you today?")

    while True:
        command = recognize_speech()
        if command:
            if "exit" in command:
                speak("Goodbye!")
                break
            elif "remember that" in command:
                remember_message = command.replace("remember that", "").replace("jarvis", "")
                speak("You told me to remember that " + remember_message)
                with open(reminder_file_path, "a") as remember_file:
                    remember_file.write(remember_message + '\n')
            elif "what do you remember" in command:
                try:
                    with open(reminder_file_path, "r") as remember_file:
                        remembered_messages = remember_file.read()
                        if remembered_messages:
                            speak("You told me to remember that " + remembered_messages)
                        else:
                            speak("I don't have any specific memory.")
                except FileNotFoundError:
                    speak("The reminder file is not found. Please check the path.")
            else:
                speak("Command not recognized. Please try again.")
