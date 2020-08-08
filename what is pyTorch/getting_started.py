from __future__ import print_function
import torch


### Tensors ###


# Create an empty tensor (old memory values)
x = torch.empty(5, 3)
print(x)

# Random tensor
x = torch.rand(5, 3)
print(x)

# Zeroes tensor of dtype long (int)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Directly enter data to tensor
x = torch.tensor([5.5, 3])
print(x)

# Create tensor based on existing tensor.
# Inherits properties of input tensor e.g. dtype, unless new values provided
# new_* methods must take in sizes
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = x.new_zeros(5, 3)
print(x)

# Override type but result has same size
x = torch.randn_like(x, dtype=torch.float)
print(x)

x = torch.zeros_like(x)
print(x)

# Get size
# torch.Szie is a tuple
print(x.size())


### Operations ###


# Addition (two syntaxes)
x = torch.randn_like(x, dtype=torch.float)
y = torch.rand(5, 3)
print(x)
print(y)
print(x + y)
print(torch.add(x, y))

# Addition - providing output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition - in-place
# Operations that mutate a tensor in-place are post-fixed with an _
# E.g. x.copy_(), x.t_() will change x
y.add_(x)
print(y)

# Copy
x.copy_(y)
print(x)

# Transpose
x.t_()
print(x)

# numpy-like indexing applies too
x = torch.rand(5, 3)
print(x)
print(x[:, 1])

# Resizing using view
# Size -1 is calculated using other dimensions input
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)

print(y.size())
print(z.size())

# One element tensor - grab value as a Python number
x = torch.randn(1)
print(x)
print(x.item())










