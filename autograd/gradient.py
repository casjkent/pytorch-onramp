import torch


### Code from tensor.py ###


x = torch.ones(2, 2, requires_grad=True)

g = torch.ones_like(x)
# x.backward(gradient=g)
# print(x)
# print(x.grad)

# print(x.grad_fn)
y = x + 2
# print(y)
# print(y.grad_fn)

z = y * y * 3
out = z.mean()
# print(z, out)

a = torch.randn(2, 2)
a = ((a*3)/(a - 1))
# print(a.requires_grad)
a.requires_grad_(True)
# print(a.requires_grad)
b = (a * a).sum()
# print(b.grad_fn)


### Gradients ###


# Leaves are input tensors (leftmost nodes in net) and roots are output tensors (rightmost nodes)
# out.backward doesn't need an input tensor because out is a scalar
# backward() calculates gradient of output w.r.t each input leaf and stores gradient in the input leaf
out.backward()
# Print gradients d(out)/dx
print(x.grad)

# Vector-Jacobian product example
x = torch.randn(3, requires_grad=True)
y = x * 2
# Frobenius norm
while y.data.norm() < 1000:
	y = y * 2

print(y)

# y is non-scalar so needs gradient input to backward()
# v is derivative of cost function w.r.t output y
# Find Vector-Jacobian product = J' * v
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

# Gradient of cost function w.r.t inputs x (same as J' * v)
print(x.grad)

# Can stop tracking history by wrapping code block in "with torch.no_grad():"
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)

# Can alternatively stop tracking history by using detach
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
























