import torch
import torch.nn as nn
import torch.nn.functional as F

# nn.Module contains layers and a method forward(input) that returns the output


### Define the network ###


# Net class (derived class based on Module class, inherits functionality from Module class)
# Follows "LeNet" structure
# See diagram in folder
class Net(nn.Module):

	# Layer formats
	def __init__(self):
		super(Net, self).__init__()
		# Expected input size is 32x32
		# 1 input image channel, 6 output channels, 3x3 square convolution kernel
		self.conv1 = nn.Conv2d(1, 6, 3)
		# 6 input image channel, 16 output channels, 3x3 square convolution kernel
		self.conv2 = nn.Conv2d(6, 16, 3)
		# "Feed-forward" (nodes web) linear layers
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * 6 * 6, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	# Have to define the forward function
	# Implement the forward propagation (chug each layer)
	def forward(self, x):
		# Implement conv 1 then ReLU then max pooling over a 2x2 window (subsampling)
		# INPUT to S2
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# Implement conv 2 then ReLU then max pooling over a 2x2 window (subsampling)
		# S2 to S4
		# If the size is square you can specify just a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		# Resize the S4 layer to 1D vector
		x = x.view(-1, self.num_flat_features(x))
		# Linear web then ReLU
		# S4 to F5
		x = F.relu(self.fc1(x))
		# Linear web then ReLU
		# F5 to F6
		x = F.relu(self.fc2(x))
		# F6 to OUTPUT
		x = self.fc3(x)
		return x

	# Find the size of the input
	def num_flat_features(self, x):
		# All dimensions except the batch dimension
		# E.g. batch of 25 images (25x3x256x256) --> (3x256x256)
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

net = Net()
print(net)

# Learnable parameters are returned by net.parameters()
params = list(net.parameters())
print(params)
print(len(params))
# Size of conv1 weights
print(params[0].size())

# Try a random 32x32 input
# torch.nn only supports mini-batches, not a single sample
# Input format: (nSamples x nColourChannels x Height x Width)
# If you have a single input, use input.unsqueeze(0) to add a fake batch dimension
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# Zero the gradient buffers of all params
net.zero_grad()
# Backprop with random gradients
out.backward(torch.randn(1, 10))


### Loss Function ###


# Still need to compute the loss and update the weights of the network
# nn.MSEloss computes the mean squared error btwn output and target
output = net(input)
print(output.size())
# Dummy target for example
target = torch.randn(10)
print(target.size())
# resize target to match output
target = target.view(1, -1)
print(target.size())

# Find loss
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

# Follow a few steps backwards
# MSELoss
print(loss.grad_fn)
# Linear
print(loss.grad_fn.next_functions[0][0])
# ReLU
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])


### Backprop ###


# To backpropagate the error just do loss.backward(). But you need to clear the existing gradients, otherwise gradients will be accumulated to existing gradients
net.zero_grad()
print('conv1.bias.grad before backward()')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward()')
print(net.conv1.bias.grad)


### Update weights ###


# Simplest weight update rule used is Stochastic Gradient Descent (SGD):
# weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
	# .sub_() executes in-place subtraction
	f.data.sub_(learning_rate * f.grad.data)

# to impleement other update rules such as SGD, Nesterov-SGD, Adam, RMSProp etc. use torch.optim
import torch.optim as optim
# Create your optimiser with lreaning rate lr
optimiser = optim.SGD(net.parameters(), lr=0.01)

# In your training loop:
# Zero the gradient buffers each time
optimiser.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
# Perform the optimisation update
optimiser.step()


















