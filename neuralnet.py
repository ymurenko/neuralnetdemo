import torch
import torch.nn as nn
import numpy as np

input_data = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0], ])

expected_output = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [3.0],
    [2.0],
    [1.0],
    [0.0],
    [2.0],
    [2.0],
    [2.0],
    [1.0],
    [1.0], ])

# input for the neural network
X = torch.tensor(input_data, dtype=torch.float)
# output for the neural network
Y = torch.tensor(expected_output / 10, dtype=torch.float)

class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        # hidden layer 1
        self.fc1 = nn.Linear(4, 5)
        self.sigmoid = nn.Sigmoid()
        # hidden layer 2
        self.fc2 = nn.Linear(5, 5)
        self.sigmoid2 = nn.Sigmoid()
        # output layer
        self.fc3 = nn.Linear(5, 1)

    def forward(self, input):
        # pass through layer 1
        out = self.fc1(input)
        out = self.sigmoid(out)
        # pass through layer 2
        out = self.fc2(out)
        out = self.sigmoid2(out)
        # pass through output layer
        out = self.fc3(out)
        return out

# Init model
model = FeedForwardNet()
# Define the loss function
loss_function = torch.nn.MSELoss()
# define the optimizer
# (this updates the weights according to a learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for iter in range(10000):
    inputs = torch.Tensor(X)
    targets = torch.Tensor(Y)

    # Forward pass
    outputs = model(inputs)
    loss = loss_function(outputs, targets)

    # Backward pass
    optimizer.zero_grad() # Clear the last weight optimization
    loss.backward() # Compute partial derivatives with respect to each weight
    optimizer.step() # Update the weights

    if iter % 1000 == 0:
        print("iter: {}/10000, Loss: {:.8f}".format(iter, loss.item()))

# Test the model with an input it hasn't seen before
print(f"Test input: [1, 0, 0, 1]")
print("{:.4f}".format(model(torch.Tensor([1, 0, 0, 1]))[0]))
