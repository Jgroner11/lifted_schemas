import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from game import datagen_structured_perc_room

if __name__ == '__main__':
    # Data
    a, x = datagen_structured_perc_room(20) 
    D_tensor = torch.tensor(x, dtype=torch.float32)  # m x n matrix

    print(D_tensor)
    np.save("data/a.npy", a)
    torch.save(D_tensor, "data/D_tensor.pth")
