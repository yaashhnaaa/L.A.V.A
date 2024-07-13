# Jarvis Notepad Automation

Jarvis Notepad Automation is a Python script that utilizes text-to-speech, speech recognition, and GUI automation libraries to automate Notepad tasks based on user voice commands.

## Prerequisites

Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

Install the required Python libraries using the following commands:

```bash
pip install pyttsx3
pip install SpeechRecognition
pip install pyautogui
```

Note: Depending on your system, you may need to install additional dependencies, such as `Pillow` for the `pyautogui` library.

## Usage

1. Run the script by executing the following command in your terminal or command prompt:

   ```bash
   python script_name.py
   ```

   Replace `script_name.py` with the actual name of your Python script.

2. The script will start listening for voice commands. Say a command or 'exit' to stop the program.

3. Follow the instructions to perform Notepad automation tasks, such as opening Notepad, writing content, saving changes, and closing Notepad.

## Voice Commands

- **Opening Notepad:** "Open Notepad" or similar phrases.
- **Writing Content:** The script will prompt you to speak the content you want to write. You can add more content by saying 'yes' or finish by saying 'no.'
- **Saving Changes:** The script will ask if you want to save changes. Respond with 'yes' or 'no' accordingly.

## Additional Notes

- Ensure your microphone is properly set up and working.
- Adjust the time.sleep() delays in the script if needed, depending on your system's responsiveness.

