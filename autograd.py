# autograd package is the central to all neural networks in PyTorch.
# autograd.Variable is the central class of autograd

import torch
from torch.autograd import Variable

# Create a variable:
x = Variable(torch.ones(2, 2), requires_grad=True)
print x
# Operations
y = x + 2
print y
# y was created as a result of an operation, so it has a creator
print y.creator

# More operations
z = y * y * 3
out = z.mean()
print z, out

# Backpropagation:  Gradients
out.backward()	# out.backward() is equivalent to out.backward(torch.Tensor([1.0]))
print x.grad
## should print out 4.5
## o = sum(z_i) / 4, and z_i = 3 * (x_i+2)^2
## z_i = 18 when x_i=1
## do/dx_i = 3/2 *(x_i+2)
## therefore, do/dx_i=4.5

# More autograd
x = torch.randn(3)
x = Variable(x, requires_grad=True)
y = x * 2
while y.data.norm() < 1e3:
	y = y * 2
print y

gradients = torch.FloatTensor([0.1, 1.0, 1e-4])
y.backward(gradients)
print x.grad
