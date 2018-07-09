from __future__ import print_function
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

#  Given data
x_train = np.array(
    [[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59],
     [2.167], [7.042], [10.791], [5.313], [7.997], [3.1]],
    dtype=np.float32)

y_train = np.array(
    [[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53],
     [1.221], [2.827], [3.465], [1.65], [2.904], [1.3]],
    dtype=np.float32)

#  Transform Numpy to Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

#  Define W and B
w = Variable(torch.randn(1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)

#  Construct Linear Regression Model
x_train = Variable(x_train)
y_train = Variable(y_train)


def linear_model(x):
    return x * w + b


y_ = linear_model(x_train)

#  Plotting before training

plt.plot(
    x_train.data.numpy(), y_train.data.numpy(), 'bo', label='Original data')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='Estimating line')
plt.legend()
plt.show()


#  Loss
def get_loss(y_, y):
    return torch.mean((y_ - y_train)**2)


loss = get_loss(y_, y_train)
print(loss)

#  Auto gradient calculating with W & B by PyTorch
loss.backward()
print(w.grad)
print(b.grad)

#  Update one time in w, b
w.data = w.data - 1e-2 * w.grad.data
b.data = b.data - 1e-2 * b.grad.data

#  Result after update parameters
y_ = linear_model(x_train)
plt.plot(
    x_train.data.numpy(), y_train.data.numpy(), 'bo', label='Original data')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='Fitting line')
plt.legend()
plt.show()

#  Updating 100 times for the parameters
for e in range(100):
    y_ = linear_model(x_train)
    loss = get_loss(y_, y_train)

    w.grad.zero_()  # ***Note, grad should be initialized each time
    b.grad.zero_()  # ***Note, grad should be initialized each time
    loss.backward()

    w.data = w.data - 1e-2 * w.grad.data  # update w with coeff. 1e-2
    b.data = b.data - 1e-2 * b.grad.data  # update b
    print('epoch: {}, loss: {}'.format(e, loss.data[0]))

#  Result after update parameters in 100 times
y_ = linear_model(x_train)
plt.plot(
    x_train.data.numpy(), y_train.data.numpy(), 'bo', label='Original data')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='Fitting line')
plt.legend()
plt.show()