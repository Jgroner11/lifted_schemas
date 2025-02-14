import numpy as np
import torch
import pickle

from make_NN import NeuralNetwork


n = 3
k = 7

a = np.load("data/a.npy")
D_tensor = torch.load("data/D_tensor.pth", weights_only=False)

model = torch.load("models/neural_network.pth", weights_only=False)
model.eval()

x = torch.argmax(model(D_tensor), dim=1)

model_file = 'models/model3.pkl'
with open(model_file, 'rb') as f:
    chmm, _, _ = pickle.load(f)
    cost = sum(chmm.bps(x[:20], a[:20]))
    print(cost)