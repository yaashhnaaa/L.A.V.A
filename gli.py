import PySimpleGUI as sg
import pyttsx3

def speak(audio):
    engine = pyttsx3.init()
    engine.runAndWait()

def process_command(command):
    # You can implement the logic to process different commands here
    # For now, let's just return a simple response
    return "You said: " + command

layout = [
    [sg.Text("Ask Jarvis something:", font=("Arial", 12), text_color="white")],
    [sg.InputText(key='-INPUT-', font=("Arial", 12))],
    [sg.Button("Ask", font=("Arial", 12))]
]

window = sg.Window("Jarvis Voice Assistant", layout, background_color="#1e1e1e")

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == "Ask":
        user_input = values['-INPUT-']
        response = process_command(user_input)
        if response:
            speak(response)

window.close()
