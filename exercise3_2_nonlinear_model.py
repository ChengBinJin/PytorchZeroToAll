x_data = [1.0, 2.0, 3.0]
y_data = [6.0, 17.0, 34.0]

w2, w1, b = 1.0, 1.0, 0.0  # a random guess: random value
lr = 1e-2  # a learning rate
num_epochs = 100  # numbe rof epochs


# our model forward pass
def forward(x):
    return x * x * w2 + x * w1 + b


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def gradient(x, y):
    dw2 = 2 * x ** 2 * (forward(x) - y)
    dw1 = 2 * x * (forward(x) - y)
    db = 1 * (forward(x) - y)

    return dw2, dw1, db


# Before training
print('predict (before training):', 4, forward(4))

# Training loop
for epoch in range(num_epochs):
    l_ = None
    for x_val, y_val in zip(x_data, y_data):
        dw2, dw1, db = gradient(x_val, y_val)
        w2 += -lr * dw2
        w1 += -lr * dw1
        b += -lr * db
        l_ = loss(x_val, y_val)

    print("probress: {}, w2 = {}, w1 = {}, b = {}, loss = {}".format(epoch, w2, w1, b, l_))

# After training
print("predict (after training)", "4 hours:", forward(4))


