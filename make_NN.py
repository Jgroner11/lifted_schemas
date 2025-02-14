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

# Cost Function (directly on logits)
def all_one_cost(v_pred_logits, o):
    # Calculate cost to force the network to output "4" (in continuous logits space)
    m, k = v_pred_logits.shape

    v = np.zeros(k)
    v[o] = -1
    t = torch.tensor(v, dtype=torch.float32)

    ones = torch.tensor(np.ones(m), dtype=torch.float32)

    return (v_pred_logits @ t) @ ones
    # return torch.sum(v_pred_logits @ t)

def equal_distribution_cost(v_pred_logits):
    # Calculate cost to force the network to output "4" (in continuous logits space)
    m, k = v_pred_logits.shape

    ones_m = torch.tensor(np.ones(m), dtype=torch.float32)
    ones_k = torch.tensor(np.ones(k), dtype=torch.float32)

    n_each_obs =  ones_m @ v_pred_logits

    total_sum = ones_k @ n_each_obs

    # return (n_each_obs - (ones_k * m / k)) ** 2 @ ones_k
    return (n_each_obs @ n_each_obs)

def distinguish_reward_cost(v_pred_logits, D, b):


    m, k = v_pred_logits.shape

    A = np.ones((m, k))
    A[:, b:] = -1
    t = torch.tensor(A, dtype=torch.float32)
    # print(D)
    # print((2*D[:, -1]-1))
    # print(t * (2*D[:, -1]-1).view(-1, 1))

    ones_m = torch.tensor(np.ones(m), dtype=torch.float32)
    ones_k = torch.tensor(np.ones(k), dtype=torch.float32)

    return   (v_pred_logits * (t * (2*D[:, -1]-1).view(-1, 1))) @ ones_k @ ones_m


def uniquely_distinguish_cost(v_pred_logits, D, b, relativity=1):
    m, k = v_pred_logits.shape

    ones_m = torch.tensor(np.ones(m), dtype=torch.float32)
    ones_k = torch.tensor(np.ones(k), dtype=torch.float32)

    n_each_obs =  ones_m @ v_pred_logits
    total_sum = ones_k @ n_each_obs

    A = np.ones((m, k))
    A[:, b:] = -1
    t = torch.tensor(A, dtype=torch.float32)

    n_rewards = torch.sum(D[:, -1])
    ratio = (m - n_rewards) / n_rewards

    # return relativity * ((v_pred_logits * (t * (2*D[:, -1]-1).view(-1, 1))) @ ones_k @ ones_m)
    return (n_each_obs @ n_each_obs) + relativity * ((v_pred_logits * (t * ((ratio+1)*D[:, -1]-1).view(-1, 1))) @ ones_k @ ones_m)

def prob_cost(v_pred_logits, chmm, a):
    x = torch.argmax(model(D_tensor), dim=1)
    x = np.array(x, dtype=np.int64)
    # returns negative log likelihood of single sequence
    cost = chmm.bps(x, a)
    return sum(cost)

if __name__ == "__main__":
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
        loss = prob_cost(v_pred_logits, chmm, a)

        # Backward pass: calculate gradients and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    torch.save(model, "models/neural_network.pth")


    # print(D_tensor)
    precision = 3
    s = torch.round(model(D_tensor) * 10 ** precision) / (10 ** precision)
    # print(s)


