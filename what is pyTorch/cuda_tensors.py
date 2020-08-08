from __future__ import print_function
import torch
import numpy as np

x = torch.rand(5, 3)
print(x)

# Run only if CUDA is available
if torch.cuda.is_available():
	# CUDA device object (input is device type)
	device = torch.device("cuda")
	# Directly create a tensor on the GPU
	y = torch.ones_like(x, device=device)
	# Move tensor to the GPU, or just use .to("cuda")
	# .to converts either dtpye or device of the tensor input
	# The returned tensor is a copy of the input with the desired dtype and device
	x = x.to(device)
	z = x + y
	print(z)
	print(z.to("cpu", torch.double))
