import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import datasets
from torchtext.vocab import GloVe, Vocab
# from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter

# Configuración de la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenización
tokenizer = get_tokenizer('basic_english')

# Cargar el dataset de IMDb
train_data, test_data = datasets.IMDB(split=('train', 'test'))

# Preprocesamiento de datos
def preprocess_data(data):
    processed_data = []
    for label, line in data:
        tokenized_line = tokenizer(line)
        processed_data.append((tokenized_line, 1 if label == 'pos' else 0))
    return processed_data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Crear vocabulario con GloVe embeddings
glove = GloVe(name='6B', dim=100)
# print(glove.stoi)
counter = Counter(glove.stoi)
counter['<unk>'] = 1  # Agregar el token UNK
# vocab = glove.stoi
vocab = Vocab(counter)

# Asegúrate de que el índice para <unk> esté configurado correctamente
UNK_IDX = vocab['<unk>']


# Definir el modelo de análisis de sentimientos
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

# Instanciar el modelo
VOCAB_SIZE = len(vocab)-1
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = SentimentModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

# Copiar embeddings pre-entrenados al modelo
model.embedding.weight.data.copy_(glove.vectors)

# Definir la función de pérdida y el optimizador
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Función de entrenamiento
# def train_model(model, data, optimizer, criterion, batch_size):
#     model.train()
#     epoch_loss = 0
#     for i in range(0, len(data), batch_size):
#         batch = data[i:i + batch_size]
#         texts, labels = zip(*batch)
#         texts = [torch.tensor([vocab.get(word, 0) for word in text]).to(device)  # 0 for UNK words
#                 for text in texts] 
#                #[torch.tensor([vocab[word] for word in text]).to(device) 
#                # for text in texts]
#         labels = torch.tensor(labels).float().to(device)

#         optimizer.zero_grad()
#         predictions = model(torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)).squeeze(1)
#         loss = criterion(predictions, labels)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     return epoch_loss / (len(data) // batch_size)
# Función de entrenamiento
def train_model(model, data, optimizer, criterion, batch_size):
    model.train()
    epoch_loss = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        texts, labels = zip(*batch)
         #torch.tensor([vocab(word, UNK_IDX) for word in text]).to(device)  # Usa UNK para palabras desconocidas
        texts = [
            torch.tensor([vocab[word] if word in vocab else UNK_IDX for word in text]).to(device)  
            for text in texts
        ]
        labels = torch.tensor(labels).float().to(device).view(-1, 1)

        optimizer.zero_grad()
        predictions = model(torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / (len(data) // batch_size)

# Entrenamiento del modelo
N_EPOCHS = 5
BATCH_SIZE = 64  # 64 # 1043 320

for epoch in range(N_EPOCHS):
    train_loss = train_model(model, train_data, optimizer, criterion, BATCH_SIZE)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')

# Función para predecir el sentimiento
def predict_sentiment(model, text):
    model.eval()
    with torch.no_grad():
        text_tensor = torch.tensor([vocab[word] for word in tokenizer(text)]).unsqueeze(0).to(device)
        prediction = torch.sigmoid(model(text_tensor))
        return prediction.item()

# Prueba de reseñas
positive_review = "This movie is fantastic! I loved every moment of it."
negative_review = "The movie was terrible. I regretted watching it."

positive_sentiment = predict_sentiment(model, positive_review)
negative_sentiment = predict_sentiment(model, negative_review)

print(f"Positive Review Sentiment: {positive_sentiment:.3f}")
print(f"Negative Review Sentiment: {negative_sentiment:.3f}")
