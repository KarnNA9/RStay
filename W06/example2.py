import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate some synthetic sequential data for GRU
data = np.array([[i for i in range(10)], [i for i in range(1, 11)]])
target = np.array([i for i in range(2, 12)])

# Convert the data to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32).view(10, 2, 1)
target = torch.tensor(target, dtype=torch.float32)

# Define the GRU model


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).requires_grad_()

        # Forward propagate GRU
        out, _ = self.gru(x, h0.detach())

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Initialize the GRU model
input_size = 1
hidden_size = 32
num_layers = 2
output_size = 1
model = GRUModel(input_size, hidden_size, num_layers, output_size)

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

# Testing the trained GRU model
test_data = torch.tensor([[10], [11]], dtype=torch.float32).view(1, 2, 1)
with torch.no_grad():
    predictions = model(test_data)

print("\nPredictions:")
print(predictions.view(-1).numpy())
