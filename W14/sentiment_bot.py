from textblob import TextBlob

def get_sentiment(text):
    # Create a TextBlob object with the user's input
    blob = TextBlob(text)

    # Get the sentiment polarity (-1 to 1, where -1 is negative, 0 is neutral, and 1 is positive)
    sentiment_polarity = blob.sentiment.polarity

    # Determine the sentiment label based on polarity
    if sentiment_polarity < 0:
        return "negative"
    elif sentiment_polarity == 0:
        return "neutral"
    else:
        return "positive"

# Chatbot loop
print("Chatbot: Hi! I'm the Sentiment Analysis Bot. You can type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    sentiment = get_sentiment(user_input)
    print(f"Chatbot: The sentiment of your input is {sentiment}.")
