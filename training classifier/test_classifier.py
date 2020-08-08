import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


### Test the network on the test data ###


# Compose several transforms together
my_tr = transforms.Compose([transforms.ToTensor(),
							transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Load and normalise test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=my_tr)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
	# Denormalise
	img = img / 2 + 0.5
	npimg = img.numpy()
	# Swap channel order RGB
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

# get random test images
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
# Show actual labels
print('Groundtruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Redefine the network
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

# Load back our trained & saved model
net = Net()
PATH = './my_cifar_net.pth'
net.load_state_dict(torch.load(PATH))

# Test run some images
outputs = net(images)
# Get the class index of the highest energy output for each image
# Other input is the dimension to reduce
_, predicted = torch.max(outputs, 1)
# output predictions
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# Look at how it performed over whole data set
correct_sum = 0
total = 0
# Also test performance by class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
	# one batch of 4 images at a time
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		# Whole data set
		total += labels.size(0)
		correct_sum += (predicted == labels).sum().item()
		# squeeze() simplifies dimensions of the output
		correct = (predicted == labels).squeeze()
		# Classes
		# 4 images per batch
		for i in range(4):
			# Find which class to add to
			label = labels[i]
			# Add to that class if it was correctly labelled
			class_correct[label] += correct[i].item()
			class_total[label] += 1

print('Accuracy of the network on 10000 test images: %d %%' % (100 * correct_sum / total))
for i in range(10):
	print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
























