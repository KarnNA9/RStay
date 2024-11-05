import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator
from transformers import BertTokenizer, BertForSequenceClassification

# Set up Fields for data preprocessing
TEXT = Field(tokenize='spacy', lower=True,
             include_lengths=True, batch_first=True)
LABEL = LabelField(dtype=torch.float)

# Load IMDb dataset and split into train and test sets
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build vocabulary based on pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
TEXT.build_vocab(train_data, vectors=tokenizer.vocab)

# Define BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=1)
model.config.id2label = {0: 'negative', 1: 'positive'}
model.config.label2id = {'negative': 0, 'positive': 1}

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create data iterators for train and test sets
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=32,
    sort_within_batch=True,
    device=device
)

# Training the sentiment analysis model


def train_model(model, iterator, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, attention_mask=(text != 1))[0].squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(iterator)

# Evaluating the sentiment analysis model


def evaluate_model(model, iterator, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, attention_mask=(text != 1))[0].squeeze(1)
            loss = criterion(predictions, batch.label)
            total_loss += loss.item()

    return total_loss / len(iterator)


# Training loop
N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    train_loss = train_model(model, train_iterator, optimizer, criterion)
    test_loss = evaluate_model(model, test_iterator, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tTest Loss: {test_loss:.3f}')
