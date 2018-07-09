import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(2017)

#  Read from data.txt
with open('chapter3_NN/data.txt', 'r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]
#  Normalized and Dividing clusters
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]

x0 = list(filter(lambda x: x[-1] == 0.0,
                 data))  # Select 1st cluster point, labeled by 0 in data.txt
x1 = list(filter(lambda x: x[-1] == 1.0,
                 data))  # Select 2nd cluster point, labeled by 1 in data.txt

plot_x0 = [i[0] for i in x0]  # i[0] in x0 has all x position
plot_y0 = [i[1] for i in x0]  # i[1] in x0 has all y position
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]

#  Plot
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bs', label='x_1')
plt.legend(loc='best')
plt.show()

#  Transform to NumPy
np_data = np.array(data, dtype='float32')  # to numpy array
x_data = torch.from_numpy(np_data[:, 0:2])  # to Tensor, the size is [100, 2]
y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1)  # to Tensor, [100, 1]

#  Define Sigmoid function
#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))

#  Plot Sigmoid function
#plot_x = np.arange(-10, 10.01, 0.01)
#plot_y = sigmoid(plot_x)
#plt.plot(plot_x, plot_y, 'r')
#plt.show()

x_data = Variable(x_data)
y_data = Variable(y_data)

#  Construct logistic regression model
import torch.nn.functional as F
w = Variable(torch.randn(2, 1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)


def logistic_regression(x):  # Sigmoid function from torch.nn
    return F.sigmoid(torch.mm(x, w) + b)


#  Plot result before updating
w0 = w[0].data[0].numpy()
w1 = w[1].data[0].numpy()
b0 = b.data[0].numpy()

plot_x = np.arange(0.2, 1, 0.01)
#plot_x = torch.from_numpy(plot_x).float()
#plot_x = Variable(plot_x, requires_grad=True)
plot_y = (-w0 * plot_x - b0) / w1

#plt.plot( # Tensor version
#   plot_x.detach().numpy(),
#  plot_y.detach().numpy(),
# 'g',
#label='cutting line')
plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bs', label='x_1')
plt.legend(loc='best')
plt.show()


#  loss
def binary_loss(y_pred, y):  # add .clamp for preventing goes to infinite
    logits = (y * y_pred.clamp(1e-12).log() +
              (1 - y) * (1 - y_pred).clamp(1e-12).log()).mean()
    return -logits


y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred, y_data)
print(loss)

#  Update
loss.backward()
w.data = w.data - 0.1 * w.grad.data
b.data = b.data - 0.1 * b.grad.data

# Update loss after previous
y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred, y_data)
print(loss)

# Use torch.optim to update parameters
from torch import nn
w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))

optimizer = torch.optim.SGD([w, b], lr=1.)
# Update 1000 times
import time

start = time.time()
for e in range(10000):
    # forward
    y_pred = logistic_regression(x_data)
    loss = binary_loss(y_pred, y_data)  # loss
    # backward
    optimizer.zero_grad()  # ***Using optim to let the grad be zero
    loss.backward()
    optimizer.step()  # Update parameters by optim
    # Calculating the accuracy rate
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().data[0] / y_data.shape[0]
    if (e + 1) % 200 == 0:
        print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(
            e + 1, loss.data[0], acc))
during = time.time() - start
print()
print('During Time: {:.3f} s'.format(during))

# Plot after updating 1000 times
w0 = w[0].data[0].numpy()
w1 = w[1].data[0].numpy()
b0 = b.data[0].numpy()

plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x - b0) / w1

plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bs', label='x_1')
plt.legend(loc='best')
plt.show()

# Using torch loss function
criterion_nn = nn.BCEWithLogitsLoss()  # 将 sigmoid 和 loss 写在一层，有更快的速度、更好的稳定性

w_nn = nn.Parameter(torch.randn(2, 1))
b_nn = nn.Parameter(torch.zeros(1))


def logistic_reg(x):
    return torch.mm(x, w_nn) + b_nn


optimizer_nn = torch.optim.SGD([w_nn, b_nn], 1.)
y_predNN = logistic_reg(x_data)
lossNN = criterion_nn(y_predNN, y_data)
print(lossNN.data)

#  Update 10000 times
startNN = time.time()
for e in range(10000):
    # 前向传播
    y_predNN = logistic_reg(x_data)
    lossNN = criterion_nn(y_predNN, y_data)
    # 反向传播
    optimizer_nn.zero_grad()
    lossNN.backward()
    optimizer_nn.step()
    # 计算正确率
    maskNN = y_predNN.ge(0.5).float()
    accNN = (maskNN == y_data).sum().data[0] / y_data.shape[0]
    if (e + 1) % 200 == 0:
        print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(
            e + 1, lossNN.data[0], accNN))

duringNN = time.time() - startNN
print()
print('During Time: {:.3f} s'.format(duringNN))