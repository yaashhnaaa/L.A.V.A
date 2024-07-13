import openai
import speech_recognition as sr
import pyttsx3

def chat_with_assistant():
    API_KEY = open("C:\\Users\\30260\\OneDrive\\Desktop\\apikeyopenai.txt","r").read()
    openai.api_key = API_KEY.strip()

    engine = pyttsx3.init()

    # Speak "Listening..." when the program starts
    speak(engine, "Listening...")

    chat_log = []

    while True:
        # Get user speech input
        user_message = get_user_speech(engine)

        if user_message and (user_message.lower() == "quit" or user_message.lower() == "exit"):
            break
        elif user_message:
            chat_log.append({"role": "user", "content": user_message})

            # Extract content from chat_log and join into a single string
            prompt = "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_log])

            # Generate assistant response with increased max_tokens limit (e.g., 200)
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=200  # Set your desired maximum number of tokens here
            )

            assistant_response = response['choices'][0]['text']

            # Print and store the assistant's response
            print("L.A.v.A:", assistant_response.strip())
            chat_log.append({"role": "assistant", "content": assistant_response.strip()})

            # Speak the assistant's response
            speak(engine, assistant_response.strip())

    # If the loop is exited, close the program
    speak(engine, "Goodbye!")
    engine.stop()

def get_user_speech(engine):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            user_message = recognizer.recognize_google(audio)
            print("You:", user_message)
            return user_message
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Could you please repeat?")
            return None
        except sr.RequestError as e:
            print("Error with the speech recognition service; {0}".format(e))
            return None
        except sr.WaitTimeoutError:
            print("Speech recognition timed out. Please try again.")
            return None

def speak(engine, response_text):
    engine.say(response_text)
    engine.runAndWait()

# Start the chat
chat_with_assistant()
