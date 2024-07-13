import openai

# Read API key from file
API_KEY = open("D:\BACKUP C\snake games\API_key.txt", "r").read()
openai.api_key = API_KEY.strip()  # Remove leading/trailing whitespaces

chat_log = []

while True:
    user_message = input("You: ")
    
    if user_message.lower() == "quit":
        break
    else:
        chat_log.append({"role": "user", "content": user_message})
        
        # Generate assistant response
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=chat_log,
            max_tokens=150
        )
        
        assistant_response = response['choices'][0]['text']
        
        # Print and store the assistant's response
        print("L.A.v.A:", assistant_response.strip())
        chat_log.append({"role": "assistant", "content": assistant_response.strip()})
