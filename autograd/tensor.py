import torch

# Setting tensor attribute .requires_grad to True, tensor will start to track all operations on it
# To stop tensor from tracking computation history, use .detach()
# To prevent tracking history wrap the code in "with torch.no_grad():". Useful when we have trainable parameters that don't need the gradients
x = torch.ones(2, 2, requires_grad=True)

# Can call .backward() to have all gradients computed automatically. If tensor is non-scalar, need to specify a gradient argument which should be a tensor OF ONES of the same shape.
# gradient input is the derivative of the loss function  w.r.t. the output vector
# Gradient will be put into .grad attribute of all the leaf nodes (inputs) connected to the output node
# out.backward() calculates d(out)/d(leaf) for each leaf
g = torch.ones_like(x)
x.backward(gradient=g)
print(x)
print(x.grad)

# Function class also very important for autograd
# Each tensor has a .grad_fn attribute that references a Function that has created the Tensor - except for tensors created by the user, which have grad_fn is None
print(x.grad_fn)
y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

# .requires_grad_() changed a tensor's flag in-place (default value is False)
a = torch.randn(2, 2)
a = ((a*3)/(a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
