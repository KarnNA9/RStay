# RULE.py
import re                       # Regex
from collections import Counter # Count elements in list
import time

timeTest = []
timeTrain = []

# Model will classify text according to these emotions.
emotions = ['love', 'fear', 'sadness', 'surprise', 'joy', 'anger']

# Common words with to be excluded during text analysis to focus on relevant words.
stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
                 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'feeling', 'feel', 'really', 'im', 'like',
                 'know', 'get', 'ive', "im'", 'stil', 'even', 'time', 'want', 'one', 'cant', 'think', 'go', 'much', 'never', 'day', 'back', 'see', 'still', 'make', 'thing',
                 'would', "would'", "could'", 'little',])

# Create a list of the most common words associated with each emotion from the training file file_path.
def create_lexicon(file_path, counter_most_common):
    # Create a dictionary where each emotion is associated with a Counter that will accumulate the most frequent words related to that emotion.
    emotion_counters = {emotion: Counter() for emotion in emotions}
    # Open training file. Must be in format "text;emotion"
    with open(file_path, 'r', encoding='utf-8') as file:
        # For each line, separate text from label (emotion)
        for line in file:
            text, emotion = line.strip().split(';')
            if emotion in emotions:
                # Normalize text to lowercase and use regex '\w+' to extract alphanumeric words.
                words = [word for word in re.findall(r'\w+', text.lower()) if word not in stop_words]
                # Add word array to dictionary
                emotion_counters[emotion].update(words)


    # After processing the training data, generate a lexicon per emotion using counter_most_common (most frequent words of each counter). Then, returns this lexicon
    emotion_lexicon = {emotion: [word for word, _ in counter.most_common(counter_most_common)] for emotion, counter in emotion_counters.items()}

    return emotion_lexicon

# Predict emotions from file and evaluate the accuracy of the model.
def predict(file_path, lexicon):
    # Initialize counters
    correct_predictions = 0
    total_predictions = 0

    # Read file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, actual_emotion = line.strip().split(';')
            # Normalize text to lowercase, exclude repetitions and use regex '\w+' to extract alphanumeric words
            words = set(re.findall(r'\w+', text.lower()))

            # Initializes a dictionary to accumulate scores for each emotion.
            emotion_scores = {emotion: 0 for emotion in lexicon}

            # calculate scores
            # For each word, increment the score of the corresponding emotion if the word is in the lexicon for that emotion.
            for word in words:
                for emotion, emotion_words in lexicon.items():
                    if word in emotion_words:
                        emotion_scores[emotion] += 1

            # Predict the emotion based on the highest score
            predicted_emotion = max(emotion_scores, key=emotion_scores.get)

            # Compare prediction with actual emotion
            if predicted_emotion == actual_emotion:
                # If prediction is correct, sum
                correct_predictions += 1
            total_predictions += 1

    # Calculate Accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy



if __name__ == '__main__':
    train_file_path = 'data/train.txt'
    test_file_path = 'data/test.txt'

    # Hyperparameters:
    increment_step = 10 # recommended: max_word_count/40
    max_word_count = 2000

    total_start_time = time.time() # Register start time

    # train and test
    # iterations starts 5, stops before max_word_count, increasing by increment_step
    # counter_most_common controlls the lexicon size
    print("Counter Most Common: Prediction Accuracy [%]:")
    for counter_most_common in range(10, max_word_count, increment_step):
        training_start_time = time.time() # Register Time for training
        lexicon = create_lexicon(train_file_path, counter_most_common)
        training_end_time = time.time()  # End training time

        testing_start_time = time.time() # Start time of testing
        accuracy = predict(test_file_path, lexicon)
        testing_end_time = time.time()   # End time of testing

        timeTest.append(round(testing_end_time-testing_start_time, 2))
        timeTrain.append(round(training_end_time-training_start_time, 2))

        # print(f"{counter_most_common}   {accuracy*100:.2f}")
        # lst = [counter_most_common, accuracy*100]
        # print('{:<20d} {:.2f}'.format(*lst))
        # print(f"{accuracy*100:.2f}")

    print("\ntimeTrain")
    print(*timeTrain,sep="\n")
    print("\ntimeTest")
    print(*timeTest,sep="\n")
    total_end_time = time.time() # end total time
    print(f"total time: {round(total_end_time-total_start_time, 2)}")
    print("End")