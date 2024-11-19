import text2emotion as te

def get_emotion(text):
    # Use text2emotion to get emotion from the user's input
    emotion = te.get_emotion(text)
    return max(emotion, key=emotion.get)

# Chatbot loop
print("Chatbot: Hi! I'm the Emotion Detection Bot. You can type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    emotion = get_emotion(user_input)
    print(f"Chatbot: The dominant emotion in your input is {emotion}.")
