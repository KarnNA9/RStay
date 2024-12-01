# DL.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import time

# Parameters
MAX_SEQ_LENGTH = 100  # Max length allowed for each sequence (although not explicitly used).
BATCH_SIZE = 8     # Batch size for data loading.
EMBEDDING_DIM = 300 # Dimension of word vectors.
HIDDEN_DIM = 512    # Number of hidden layers of the LSTM.
EPOCHS = 20      #

total_start_time = time.time()

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=';', header=None, names=['text', 'emotion'])
    return df['text'].values, df['emotion'].values

train_texts, train_labels = load_data('data/train.txt')
test_texts, test_labels = load_data('data/test.txt')

# Build vocabulary
word_counter = Counter()
for text in train_texts:
    word_counter.update(text.split())
vocab = ['<PAD>', '<UNK>'] + [word for word, freq in word_counter.items() if freq > 1]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab)

# Tokenize and encode labels
def tokenize(text):
    return [word_to_idx.get(word, word_to_idx['<UNK>']) for word in text.split()]

def encode_labels(labels):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(labels), label_encoder

encoded_train_labels, label_encoder = encode_labels(train_labels)
encoded_test_labels, _ = encode_labels(test_labels)

# Dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx):
        self.texts = [torch.tensor(tokenize(text)) for text in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Padding function
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=word_to_idx['<PAD>'])
    return texts, torch.tensor(labels)

# Data Loaders
train_dataset = EmotionDataset(train_texts, encoded_train_labels, word_to_idx)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

test_dataset = EmotionDataset(test_texts, encoded_test_labels, word_to_idx)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# LSTM Model
class LSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(LSTMEmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word_to_idx['<PAD>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)

# Model initialization
model = LSTMEmotionClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
train_start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print(loss.item())
train_end_time = time.time()

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels in data_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Test the model
test_start_time = time.time()
test_accuracy = evaluate_model(model, test_loader)
# print(f'Test Accuracy: {test_accuracy*100:.2f}%')
test_end_time = time.time()

# Save the model to disk
torch.save(model.state_dict(), 'lstm_emotion_classifier_model.pth')

# Load the model from disk
loaded_model = LSTMEmotionClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, len(label_encoder.classes_))
loaded_model.load_state_dict(torch.load('lstm_emotion_classifier_model.pth'))

# Validate the loaded model
val_texts, val_labels = load_data('data/val.txt')
encoded_val_labels, _ = encode_labels(val_labels)
val_dataset = EmotionDataset(val_texts, encoded_val_labels, word_to_idx)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

val_start_time = time.time()
validation_accuracy = evaluate_model(loaded_model, val_loader)
val_end_time = time.time()

total_end_time = time.time()

print(f'Test Accuracy: {test_accuracy*100:.2f}%')
print(f'Validation Accuracy: {validation_accuracy*100:.2f}%')
print(f"Training time: {(train_end_time-train_start_time):.2f}")
print(f"Testing time: {(test_end_time-test_start_time):.2f}")
print(f"Eval time: {(val_end_time-val_start_time):.2f}")
print(f"Total time: {(total_end_time-total_start_time):.2f}")
print(f"\n{test_accuracy*100:.2f}%\n{validation_accuracy*100:.2f}%\n{(train_end_time-train_start_time):.2f}\n{(test_end_time-test_start_time):.2f}\n{(val_end_time-val_start_time):.2f}\n{(total_end_time-total_start_time):.2f}\n\n\n")