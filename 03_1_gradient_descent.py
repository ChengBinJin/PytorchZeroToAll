x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value
lr = 0.01  # learning rate
num_epochs = 100  # number of epochs


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def gradient(x, y):
    return 2 * x * (x * w - y)


# Before training
print("predict (before training):", 4, forward(4))

# Training loop
for epoch in range(num_epochs):
    l_ = None
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - lr * grad
        print("\tgrad: ", x_val, y_val, grad)
        l_ = loss(x_val, y_val)

    print("progrss:", epoch, "w=", w, "loss=", l_)

# After training
print("predict (after training)", "4 hours:", forward(4))



