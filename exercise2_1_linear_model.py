import csv
import numpy as np
import matplotlib.pyplot as plt

data = []
with open('diabetes.csv', 'r') as csvfile:
    readers = csv.reader(csvfile, delimiter=',')
    for row in readers:
        data.append(row)

data = np.asarray(data).astype(np.float32)
num_data = data.shape[0]
print('number of data: {}'.format(num_data))

x_data, y_data = data[:, :-1], data[:, -1]
print('x_data shape: {}'.format(x_data.shape))
print('y_data shape: {}'.format(y_data.shape))


# normalize x_data
mean = np.mean(x_data, axis=0)
std = np.std(x_data, axis=0)
print('mean shape {}, mean: {}'.format(mean.shape, mean))
print('std shape {}, std: {}'.format(std.shape, std))
x_data = (x_data - mean) / std


# our model for the forward pass
def forward(X):
    w = np.random.randn(x_data.shape[1], 1)
    return np.dot(X, w)


def loss(y_pred, y):
    return (y_pred - y) * (y_pred - y)


x_list, mse_list = [], []
for itr in range(100):
    l_sum = 0.
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l_ = loss(y_pred_val, y_val)
        l_sum += l_
        # print('x_val: {},\ny_val: {},\ny_pred_val: {},\nloss: {}\n'.format(x_val, y_val, y_pred_val, l_))

    print('iter: {}, MSE = {}'.format(itr, l_sum / num_data))

    x_list.append(itr)
    mse_list.append(l_sum / num_data)

plt.plot(x_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('Try')
plt.show()

