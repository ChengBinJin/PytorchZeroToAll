import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 0.0  # a random guess: random value, 0.0


# our model for the forward pass
def forward(x):
    return x * w


# Loss function
def loss(y_pred, y):
    return (y_pred - y) * (y_pred - y)


w_list = []
mse_list = []
num = len(x_data)
for w in np.arange(0.0, 4.1, 0.1):
    print("w = {:.3f}".format(w))
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l_ = loss(y_pred_val, y_val)
        l_sum += l_
        print("\t", x_val, y_val, y_pred_val, l_)

    print("MSE = {:.3f}\n".format(l_sum / num))

    w_list.append(w)
    mse_list.append(l_sum / num)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()

