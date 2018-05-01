x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 5.0
lr = 0.1


def forward(x):
    return x * w


def loss(y, y_pred):
    return (y - y_pred) * (y - y_pred)


def gradient(x, y, y_pred):
    return 2. * x * (y_pred - y)


print('Before training: 4 hours - {} scores'.format(forward(4)))

for iter_ in range(100):
    loss_ = 0.
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)
        loss_ += loss(y_val, y_pred)
        w += -lr * gradient(x_val, y_val, y_pred)

    print('iter: {}, loss: {}'.format(iter_, loss_))

print('After training: 4 hours - {} scores'.format(forward(4)))

