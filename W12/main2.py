import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# Definir el dispositivo (GPU si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el tokenizer de BERT (o cualquier otro modelo de transformers)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Definir el modelo de clasificación
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=1)
model.to(device)

# Cargar el dataset de IMDb de torchtext (versión más reciente)
train_data, test_data = IMDB(split=('train', 'test'))

# Tokenización y preparación de los datos
def tokenize_data(example):
    return tokenizer(
        example['text'],
        truncation=True,
        padding='max_length',  # Cambiar a True si se usa DataCollatorWithPadding
        max_length=512,
        return_tensors='pt'
    )

# Tokenizar y preparar los datos con DataLoader
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(train_data.map(tokenize_data), batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data.map(tokenize_data), batch_size=32, shuffle=False, collate_fn=collate_fn)

# Definir la función de pérdida y el optimizador
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Funciones de entrenamiento y evaluación
def train_model(model, iterator, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in iterator:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.squeeze(1)
        loss = criterion(predictions, labels.float())
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(iterator)

def evaluate_model(model, iterator, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.squeeze(1)
            loss = criterion(predictions, labels.float())
            total_loss += loss.item()

    return total_loss / len(iterator)

# Bucle de entrenamiento
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_loss = train_model(model, train_loader, optimizer, criterion)
    test_loss = evaluate_model(model, test_loader, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tTest Loss: {test_loss:.3f}')
