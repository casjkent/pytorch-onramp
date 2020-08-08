# Use standard python packages to read data into a numpy array, then convert into a torch.*Tenspor
# images - OpenCV/torchvision
# audio - scipy librosa
# text - raw Python, NLTK or SpaCy

NUM_EPOCHS = 2
NUM_BATCHES = 200


### Training an image classifier ###


# Using the CIFAR10 data set (3x32x32). 50000 training data and 10000 test data.
# Load and normalise the CIFAR10 data set using torchvision
import torch
import torchvision
# Contains common image transformation tools
import torchvision.transforms as transforms

# Output of torchvision datasets are PILImage (Pyton Imaging Library) images of range [0,1]. We transform them to Tensors of normalised range [-1,1].
# Compose several transforms together
# torchvision.transforms.ToTensor() converts a PIL image or numpy.ndarray to tensor.
# norchvision.transforms.Normalise() normalises each channel of a Tensor image. Takes in (mean[1], ..., mean[n]) and (std[1], ..., std[n]) for n channels (in this case 3)
# This is not an in-place normalisation
my_tr = transforms.Compose([transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load and normalise training dataset
# root is path to dataset where cifar-10-batches-py exists or will be saved if download is set to True
# train=True - choose either training dataset or test dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=my_tr)
# shuffle - shuffle at every epoch
# num_workers - number of subprocesses to use for data loading
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# Load and normalise test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=my_tr)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# let's show some training images
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
	# Denormalise
	img = img / 2 + 0.5
	npimg = img.numpy()
	# Swap channel order RGB
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

# get random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Make a grid of images from the batch and show images
imshow(torchvision.utils.make_grid(images))
# Print labels to screen
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


### Define a convolutional neural network ###


# Copied from neural_net.py, modified to take 3-channel images instead of 1-channel
import torch.nn as nn
import torch.nn.functional as F

# nn.Module contains layers and a method forward(input) that returns the output
# Net class (derived class based on Module class, inherits functionality from Module class)
# Follows "LeNet" structure
# See diagram in folder
class Net(nn.Module):

	# Layer formats
	def __init__(self):
		super(Net, self).__init__()
		# Expected input size is 32x32
		# 3 input image channels, 6 output channels, 5x5 square convolution kernel
		self.conv1 = nn.Conv2d(3, 6, 5)
		# Max pooling 2x2
		self.pool = nn.MaxPool2d(2, 2)
		# 6 input image channel, 16 output channels, 5x5 square convolution kernel
		self.conv2 = nn.Conv2d(6, 16, 5)
		# "Feed-forward" (nodes web) linear layers
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	# Have to define the forward function
	# Implement the forward propagation (chug each layer)
	def forward(self, x):
		# Implement conv 1 then ReLU then max pooling over a 2x2 window (subsampling)
		# INPUT to S2
		x = self.pool(F.relu(self.conv1(x)))
		# Implement conv 2 then ReLU then max pooling over a 2x2 window (subsampling)
		# S2 to S4
		x = self.pool(F.relu(self.conv2(x)))
		# Resize the S4 layer to 1D vector
		x = x.view(-1, 16 * 5 * 5)
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


### Define loss function and optimiser ###


import torch.optim as optim

# Using Classification Cross-Entropy loss and SGD with momentum
# Cross-Entropy pounishes confident and wrong answers very strongly using a log shaped cost function
# Momentum takes moving average of gradient to eliminate noise in SGD
# 0.9 is typically a good momentum value
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


### Train the network on GPU ###


# Move to CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Move net to GPU
net.to(device)

# Loop over dataset multiple times
for epoch in range(NUM_EPOCHS):
	print('Begin epoch %d' % (epoch + 1))
	running_loss = 0.0
	# enumerate() grabs one element at a time with counter included, input is starting index of
	# counter
	for batch, data in enumerate(trainloader, 0):
		# Get the inputs
		# Remeber you have to send the inputs and targets at every step to GPU too
		inputs, labels = data[0].to(device), data[1].to(device)

		# Zero the param gradients
		optimiser.zero_grad()

		# Forward + backprop + optimise
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimiser.step()

		# print stats
		running_loss += loss.item()
		# Every NUM_BATCHES mini-batches
		if batch % NUM_BATCHES == (NUM_BATCHES - 1):
			print('[epoch, mini-batch]: [%d, %4d] loss: %.3f' % (epoch + 1, batch + 1, \
				  running_loss / NUM_BATCHES))
			running_loss = 0.0

print('Finished training! :)')


### Save the trained model ###


PATH = './my_cifar_net.pth'
torch.save(net.state_dict(), PATH)
print('saved model to %s' % PATH)




















