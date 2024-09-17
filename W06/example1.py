import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate some synthetic sequential data for LSTM
data = np.array([[i for i in range(10)], [i for i in range(1, 11)]])
target = np.array([i for i in range(2, 12)])

# print(data)
# print(target)

# Convert the data to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32).view(2, 10, 1) # view(2, 10, 1)
target = torch.tensor(target, dtype=torch.float32).view(1, 10, 1) # view included 

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Initialize the LSTM model
input_size = 1
hidden_size = 32
num_layers = 2
output_size = 1
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Define the loss function (Mean Squared Error) and the optimizer (Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs.view(-1), target)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Testing the trained LSTM model
test_data = torch.tensor([[10], [11]], dtype=torch.float32).view(1, 2, 1)
with torch.no_grad():
    predictions = model(test_data)

print("\nPredictions:")
print(predictions.view(-1).numpy())
