x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 10.0, 21.0]

# answer x^2 * w2 + x * w1 + b
# w2 = 2, w1 = 1, b = 0

num_iters = 100
lr = 1e-2
w2 = w1 = b = 3.0  # random guess


def forward(x):
    return x * x * w2 + x * w1 + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


def dloss_by_dypred(x, y):
    return 2 * (forward(x) - y)


def dypred_by_dw2(x):
    return x * x


def dypred_by_dw1(x):
    return x


def dypred_by_db():
    return 1


def dloss_by_dw1(x, y):
    return dloss_by_dypred(x, y) * dypred_by_dw1(x)


def dloss_by_dw2(x, y):
    return dloss_by_dypred(x, y) * dypred_by_dw2(x)


def dloss_by_db(x, y):
    return dloss_by_dypred(x, y) * dypred_by_db()


print('Before training, 4 hours: {}, correct answer should be 36!'.format(forward(4.0)))

for iter_time in range(num_iters):
    loss_ = 0.
    for x_val, y_val in zip(x_data, y_data):
        loss_ += loss(x_val, y_val)

        w2 += -lr * dloss_by_dw2(x_val, y_val)
        w1 += -lr * dloss_by_dw1(x_val, y_val)
        b += -lr * dloss_by_db(x_val, y_val)

    print(iter_time, 'loss: {}'.format(loss_/len(x_data)))

print('After training, 4 hours: {}, correct answer should be 36!'.format(forward(4.0)))

