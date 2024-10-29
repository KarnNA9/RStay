import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import datasets
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader

# Tokenization
tokenizer = lambda x: x.split()

# Load IMDb dataset
train_data, test_data = datasets.IMDB()

# Create vocabulary with GloVe embeddings
glove = GloVe(name='6B', dim=100)
vocab = glove.stoi

# Define the sentiment analysis model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden[-1])

# Instantiate the model
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = SentimentModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# Copy pre-trained embeddings to model
model.embedding.weight.data.copy_(glove.vectors)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())


# Training the model
def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train_loss = train_model(model, train_iterator, optimizer, criterion)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')

# Testing the model


def predict_sentiment(model, text):
    model.eval()
    with torch.no_grad():
        text = torch.tensor([TEXT.vocab.stoi[word]
                            for word in text]).unsqueeze(1).to(device)
        prediction = torch.sigmoid(model(text))
        return prediction.item()


# Test a positive review
positive_review = "This movie is fantastic! I loved every moment of it."
positive_sentiment = predict_sentiment(model, positive_review)
print(f"Positive Review Sentiment: {positive_sentiment:.3f}")

# Test a negative review
negative_review = "The movie was terrible. I regretted watching it."
negative_sentiment = predict_sentiment(model, negative_review)
print(f"Negative Review Sentiment: {negative_sentiment:.3f}")