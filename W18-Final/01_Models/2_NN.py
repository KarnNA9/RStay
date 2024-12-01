import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
import time

# Hyperparameters
MAX_SEQ_LENGTH = 50 # Max length of each text sequence.
EPOCHS = 100         # Num of complete Iiterations on the training data.
EMBED_DIM = 300     # Dimension of embedding vectors for words.
                    # ~ Input dimension of the layer. 

# Tokenize and pad/truncate
def tokenize(text, max_length):
    # Normalize text (only alphanumeric and lowercasing)
    tokens = re.findall(r'\w+', text.lower())
    # All sequences MUST have the same length (
    if len(tokens) > max_length:
        return tokens[:max_length]
    return tokens + ['<PAD>'] * (max_length - len(tokens))

# Read data from file. Return an array for text and other for labels
def load_data(file_path, max_length):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().split(';')
            texts.append(tokenize(text, max_length))
            labels.append(label)
    return texts, labels

# Prepare Dataset
class EmotionDataset(Dataset):
    # Converts texts into vocabulary-based indexes (word_to_idx) and saves tags.
    def __init__(self, texts, labels, word_to_idx):
        self.texts = [[word_to_idx.get(word, word_to_idx['<UNK>']) for word in text] for text in texts]
        self.labels = labels
    # retur dataset size
    def __len__(self):
        return len(self.texts)
    # get item as a tensor and label
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, EMBED_DIM, num_classes):
        super(EmotionClassifier, self).__init__()
        # Layer that converts word indices into embedding vectors
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        # define linear layer. Each processed text generates a feature vector of dimension EMBED_DIM by averaging the embeddings of the words.
        # num_classes -> num of output layer        
        self.fc = nn.Linear(EMBED_DIM, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        # Averages the embeddings of words in a sequence.
        x = torch.mean(x, dim=1)
        return self.fc(x)

total_start_time = time.time() # Register start time

# Load and process data
train_texts, train_labels = load_data('data/train.txt', MAX_SEQ_LENGTH)
test_texts, test_labels = load_data('data/test.txt', MAX_SEQ_LENGTH)

# Encode labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Build vocabulary
all_words = [word for text in train_texts for word in text]
word_counts = Counter(all_words) # Repetitions of each words
vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.items() if count > 1] # list of words w at least 2 rep.
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Model parameters
vocab_size = len(vocab)
num_classes = len(label_encoder.classes_)

# Model, Loss, and Optimizer
model = EmotionClassifier(vocab_size, EMBED_DIM, num_classes) # The Model
criterion = nn.CrossEntropyLoss() # Cross-entropy loss function for classification.
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer with learning rate (lr) of 0.001.

# Data loaders
# Iterators that handle batches of data. 
train_dataset = EmotionDataset(train_texts, train_labels, word_to_idx) 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # shuffle=True shuffles the data for training.

test_dataset = EmotionDataset(test_texts, test_labels, word_to_idx)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
train_start_time = time.time() # Register Time for training
for epoch in range(EPOCHS):
    for texts, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels) # Difference from model's predictions from actual values.
        loss.backward()
        optimizer.step()
    # print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print(loss.item())
train_end_time = time.time()  # End training time

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    test_start_time = time.time() # Start time of testing
    for texts, labels in test_loader: 
        outputs = model(texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_end_time = time.time() # End time of testing

accuracy2 = correct / total
print(f'Accuracy [testing]: {accuracy2*100:.2f}%')

# Save the model to disk
torch.save(model.state_dict(), 'emotion_classifier_model.pth')

# Load the model from disk
loaded_model = EmotionClassifier(vocab_size, EMBED_DIM, num_classes)
loaded_model.load_state_dict(torch.load('emotion_classifier_model.pth'))

# Function to evaluate model
def evaluate_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for texts, labels in data_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Load and process validation data
val_texts, val_labels = load_data('data/val.txt', MAX_SEQ_LENGTH)
val_labels = label_encoder.transform(val_labels)
val_dataset = EmotionDataset(val_texts, val_labels, word_to_idx)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Evaluate the loaded model on validation data
eval_start_time = time.time() # start Time for eval
accuracy = evaluate_model(loaded_model, val_loader)
eval_end_time = time.time() # end Time for eval

total_end_time = time.time()

print(f'Validation Accuracy: {accuracy*100:.2f}%')
print(f"Training time: {(train_end_time-train_start_time):.2f}")
print(f"Testing time: {(test_end_time-test_start_time):.2f}")
print(f"Eval time: {(eval_end_time-eval_start_time):.2f}")
print(f"Total time: {(total_end_time-total_start_time):.2f}")
print(f"{accuracy2*100:.2f}%\n{accuracy*100:.2f}%\n{(train_end_time-train_start_time):.2f}\n{(test_end_time-test_start_time):.2f}\n{(eval_end_time-eval_start_time):.2f}\n{(total_end_time-total_start_time):.2f}\n\n\n")
