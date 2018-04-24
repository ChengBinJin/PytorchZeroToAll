import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

lr = 0.01
num_epochs = 10

w = Variable(torch.FloatTensor([1.0]), requires_grad=True)  # Any random


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# Before training
print("predict (before training)", 4, forward(4).data[0])

# Training loop
for epoch in range(num_epochs):
    l_ = None
    for x_val, y_val in zip(x_data, y_data):
        l_ = loss(x_val, y_val)
        l_.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - lr * w.grad.data

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print('progress:', epoch, l_.data[0])

# After training
print("predict (after training)", 4, forward(4).data[0])
