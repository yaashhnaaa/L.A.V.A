import pprint
import google.generativeai as palm

palm.configure(api_key='AIzaSyAnIJ3-KA7HUGHjaHVQfJnH9fHY1RH0K9U')

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name  # We use the model without printing its name

first_input = True  # This flag tracks whether it's the first user input

while True:
    # Greet the user based on the flag
    if first_input:
        user_prompt = input("Hello user, I am L.A.V.A and what would you like to know about? ")
        first_input = False
    else:
        user_prompt = input("Now, what else would you like to know about? ")

    # Handle greetings and goodbyes
    if user_prompt.lower() in ["hello", "hi"]:
        # Check if there's a previous prompt
        if previous_prompt:
            # Use previous prompt in the greeting
            print(f"Hi there! I remember you were curious about {previous_prompt}. Is there anything else you'd like to know?")
        else:
            print("Hi there! I'm happy to help you learn about anything you'd like.")
        continue
    elif user_prompt.lower() == "goodbye":
        print("It was nice talking to you! Have a great day!")
        break

    # Update previous prompt
    previous_prompt = user_prompt

    # Generate text based on user prompt
    completion = palm.generate_text(
        model=model,
        prompt=user_prompt,
        temperature=0.5,  # Adjust temperature for desired randomness
        max_output_tokens=800,
    )

    # Print the response
    print(f"{completion.result}")

print("Thanks for using L.A.V.A!")
