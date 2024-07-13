import openai
API_KEY = open("API key","r").read()
openai.api_key = API_KEY

chat_log = []
while True:
    user_message = input()
    if user_message.lower() =="quit":
        break
    else:
        chat_log.append({"role": "user", "content": user_message})
        response = openai.ChatComplrtion.create(
            mode="gpt-3.5-turbo-1106",
            messages=chat_log
        )
        assistant_response = response['choices'][0]['message']['content']
        print("L.A.v.A:",assistant_response.strip("\n").strip())
        chat_log.append({"role":"assistant","content":assistant_response.strip("\n").strip()})
