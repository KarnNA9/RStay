import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# Define a simple feedforward neural network for emotion classification
class EmotionClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)

# Create an instance of the emotion classifier
hidden_size = 768
num_classes = 4  # 4 emotions: happy, sad, angry, neutral
model = EmotionClassifier(hidden_size, num_classes)

# Define a function to convert text to input features for BERT
def convert_text_to_features(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    return inputs['input_ids'], inputs['attention_mask']

# Example sentences to test emotion detection
sample_sentences = [
    "I am so happy today!",
    "I feel really sad about what happened.",
    "That made me angry!",
    "It's just another regular day."
]

# Convert example sentences to BERT input features and predict emotions
with torch.no_grad():
    for sentence in sample_sentences:
        input_ids, attention_mask = convert_text_to_features(sentence)
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1).tolist()[0]
        emotions = ['Happy', 'Sad', 'Angry', 'Neutral']
        max_prob_index = probabilities.index(max(probabilities))
        predicted_emotion = emotions[max_prob_index]
        print(f"Sentence: {sentence}")
        print(f"Predicted Emotion: {predicted_emotion}")
        print()
