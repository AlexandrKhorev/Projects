import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def relu(x):
    return np.maximum(x, 0)


def probability(z):
    out = np.exp(z)
    return out / np.sum(out)


def MSE_loss(y_pred, y_train):
    return ((y_train - y_pred) ** 2).mean()


S_input = 4  # n - количество входных данных
S_neuron = 15  # k - количестов нейронов в первом слое
S_output = 3  # l - количество выходов

W1 = np.random.randn(S_input, S_neuron)
W2 = np.random.randn(S_neuron, S_output)
print(W1, W2)
B1 = np.zeros((1, S_neuron))
B2 = np.zeros((1, S_output))

def predict(data):
    h1 = data @ W1 + B1
    # O1 = relu(h1)
    O1 = sigmoid(h1)

    h2 = O1 @ W2 + B2
    # O2 = probability(h2)
    O2 = sigmoid(h2)

    return O2


def training(x_train_data, y_train_data):

    global W1
    global W2
    global loss
    loss = np.zeros(len(x_train_data))

    for number_train in range(len(x_train_data)):
        x_train = x_train_data[[number_train], :]
        y_train = y_train_data[[number_train], :]
        h_pred = (sigmoid(x_train @ W1))
        y_pred = predict(x_train)

        dLdy = -2 * (y_train - y_pred)

        dW2 = h_pred.T @ (dLdy * sigmoid_deriv(y_pred))
        dW1 = x_train.T @ (((dLdy * sigmoid_deriv(y_pred)) @ W2.T) * sigmoid_deriv(h_pred))

        W1 += dW1
        W2 += dW2
        loss[number_train] = MSE_loss(y_pred, y_train)


x_train_data = np.array([[5.1, 3.5, 1.4, 0.2],
                         [4.9, 3.1, 1.5, 0.1],
                         [5.0, 3.5, 1.3, 0.3],
                         [5.1, 3.5, 1.4, 0.3],
                         [5.4, 3.7, 1.5, 0.2],
                         [5.0, 3.4, 1.5, 0.2],
                         [4.7, 3.2, 1.6, 0.2],
                         [4.9, 3.6, 1.4, 0.1],
                         [6.3, 3.3, 4.7, 1.6],
                         [6.2, 2.2, 4.5, 1.5],
                         [5.6, 2.5, 3.9, 1.1],
                         [5.9, 3.0, 4.2, 1.5],
                         [5.8, 2.6, 4.0, 1.2],
                         [6.3, 2.5, 4.9, 1.5],
                         [6.0, 2.7, 5.1, 1.6],
                         [6.4, 2.9, 4.3, 1.3],
                         [6.5, 3.0, 5.5, 1.8],
                         [6.7, 3.3, 5.7, 2.1],
                         [7.2, 3.2, 6.0, 1.8],
                         [6.4, 2.8, 5.6, 2.1],
                         [6.8, 3.2, 5.9, 2.3],
                         [6.4, 2.7, 5.3, 1.9],
                         [7.2, 3.2, 6.0, 1.8],
                         [6.4, 2.8, 5.6, 2.2]])

y_train_data = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [0, 0, 1],
                         [0, 0, 1],
                         [0, 0, 1],
                         [0, 0, 1],
                         [0, 0, 1],
                         [0, 0, 1],
                         [0, 0, 1]])

training(x_train_data, y_train_data)

x1 = [5.6, 2.8, 4.9, 2.0]       # virginica
y = ["setosa", "versicolor", "virginica"]
prediction = y[np.argmax(predict(x1))]

# plt.plot(loss)
# plt.show()

print(prediction)

