import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from game import datagen_structured_perc_room

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Linear layer
        self.softmax = nn.Softmax(dim=1)           # Softmax along the feature dimension

    def forward(self, x):
        logits = self.fc(x)  # Compute logits
        probabilities = self.softmax(logits * 10)  # Apply a very hard softmax 
        return probabilities


class CustomLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_pred_logits):
        # Save for backward pass
        x = torch.argmax(model(D_tensor), dim=1)
        x = np.array(x, dtype=np.int64)
        cost = chmm.bps(x, a)
        return sum(cost)

    @staticmethod
    def backward(ctx, grad_output):
        v_pred_logits, = ctx.saved_tensors
        m, k = v_pred_logits.shape
        ones_m = torch.ones(m, dtype=torch.float32, device=v_pred_logits.device)
        ones_k = torch.ones(k, dtype=torch.float32, device=v_pred_logits.device)
        
        n_each_obs = ones_m @ v_pred_logits
        
        # Derivative of loss w.r.t. v_pred_logits
        grad = 2 * grad_output * ones_m.unsqueeze(1) * n_each_obs.unsqueeze(0)
        return grad


# Use the custom loss in the training loop
def custom_cost(v_pred_logits):
    return CustomLossFunction.apply(v_pred_logits)


n = 3
k = 6

# Data
a = np.load("data/a.npy")
D_tensor = torch.load("data/D_tensor.pth")
with open('models/model3.pkl', 'rb') as f:
    chmm, _, _ = pickle.load(f)

model = NeuralNetwork(n, k)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass: predict logits
    v_pred_logits = model(D_tensor)  # Continuous logits

    # Calculate custom loss
    loss = custom_cost(v_pred_logits)

    # Backward pass: calculate gradients and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

torch.save(model, "models/neural_network.pth")

precision = 3
s = torch.round(model(D_tensor) * 10 ** precision) / (10 ** precision)
print(s)
