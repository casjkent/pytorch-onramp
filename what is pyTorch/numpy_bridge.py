from __future__ import print_function
import torch


### Converting torch tensor to numpy array ###


t = torch.ones(5)
print(t)

n = t.numpy()
print(n)

# Torch and numpy will share the memory locations, changing one will change the other
t.add_(1)
print(t)
print(n)


### Converting numpy array to torch tensor  ###


import numpy as np
n = np.ones(5)
t = torch.from_numpy(n)
print(n)
print(t)

# Torch and numpy will share the memory locations, changing one will change the other
np.add(n, 1, out=n)
print(n)
print(t)

# CharTensor does not support converting to numpy and back.
