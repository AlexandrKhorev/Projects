import numpy as np
import matplotlib.pyplot as plt
import time

# np.random.seed(112412)
def relu(x):
    return np.maximum(x, 0)


def deriv_relu(x):
    if x <= 0:
        return 0
    return 1


def probability(z):
    out = np.exp(z)
    print(out)
    return out / np.sum(out)


def nonlin(x, deriv=False):
    if deriv:
        return nonlin(x) * (1 - nonlin(x))
    return 1 / (1 + np.exp(-x))


def nonlin_deriv(x):
    return x * (1 - x)


def generator():
    x = np.round(np.sort(np.abs(np.random.normal(0, 1, size=4))), 2)
    # x = np.random.randint(0, 100, size=4)
    y = np.array([x[:2].mean(), x[2:].mean()])
    return x, y


def init_weights(n, m):
    """ инициализирует случайные веса в промежутке -0.5 - 0.5
    в матрице размера n,m  , где n - количество нейронов в предыдущем слое
    m - количество нейронов в следующем слое
    """
    weights = np.random.sample((n, m)) - 0.5
    return weights


def construct(inputs_number1, neuron_sloys1):
    inputs_number = np.copy(inputs_number1)
    inputs_number += 1

    neuron_sloys = np.copy(neuron_sloys1)
    neuron_sloys += 1

    total_weights = [init_weights(inputs_number, neuron_sloys[0] - 1)]

    for i in range(len(neuron_sloys) - 1):
        total_weights.append(init_weights(neuron_sloys[i], neuron_sloys[i + 1] - 1))

    return total_weights


def forward_prop(x_inputs, total_weights, inputs_number1, Neuron_layer1):
    input_number = np.copy(inputs_number1)

    x_inputs = np.hstack((x_inputs, [-1]))
    Neuron_layer1 = np.hstack(([input_number], Neuron_layer1))

    reactions = [np.array(x_inputs)]

    for i in range(len(Neuron_layer1) - 1):
        Neuron_layer = np.copy(Neuron_layer1)
        vector_reactions = np.zeros(Neuron_layer[i + 1])

        Neuron_layer[i] += 1

        for j in range(Neuron_layer[i + 1]):
            for k in range(Neuron_layer[i]):
                vector_reactions[j] += reactions[i][k] * total_weights[i][k][j]
        # vector_reactions = nonlin(vector_reactions)

        vector_reactions = relu(vector_reactions)

        if i != len(Neuron_layer1) - 2:
            vector_reactions = np.hstack((vector_reactions, [-1]))

        reactions.append(vector_reactions)

    return reactions


def structure_network():
    plt.figure()
    number_layer = len(Neuron_sloys)
    for i in range(number_layer + 1):
        for j in range(len(react[i])):
            if i != number_layer:
                # Связи

                scope = len(react[i + 1]) - 1
                if i == number_layer - 1:
                    scope = len(react[i + 1])

                for k in range(scope):
                    plt.plot([i, i + 1], [len(react[i]) / 2 - j, len(react[i + 1]) / 2 - k],
                             color='black',
                             alpha=abs(total_weight[i][j][k]) * 2,
                             linewidth=0.5)

            # Нейроны
            cc = i / (number_layer + 1) * 0.8
            color = (1 - cc, 0, cc)
            plt.plot(i, len(react[i]) / 2 - j,
                     marker='o', markersize=23, markerfacecolor="white", markeredgewidth=1.4, color=color)

            plt.text(i, len(react[i]) / 2 - j, round(react[i][j - len(react[i])], 3), horizontalalignment='center',
                     verticalalignment='center', fontsize=6.5, fontdict={'color': 'black'})
    plt.show()


def MSE_loss(y_train, y_pred):
    return (y_train - y_pred) ** 2


def MSE_loss2(y_train, y_pred):
    return - 2 * (y_train - y_pred)


def backprop(total_weights, test_case_inputs, test_case_outputs, Neuron_sloys):
    learn_rate = 0.01
    reactions = forward_prop(test_case_inputs, total_weights, len(test_case_inputs), Neuron_sloys)

    deltas = []
    # deltas.append(np.array(MSE_loss(test_case_outputs, reactions[-1])))
    deltas.append(np.array(MSE_loss2(test_case_outputs, reactions[-1])))

    if len(Neuron_sloys) > 1:
        for i in range(len(Neuron_sloys) - 2, -1, -1):

            vector_deltas = np.zeros(Neuron_sloys[i])

            for j in range(Neuron_sloys[i]):
                for k in range(Neuron_sloys[i + 1]):
                    vector_deltas[j] += deltas[0][k] * total_weights[i + 1][j][k]
            deltas.insert(0, np.array(vector_deltas))

    Neuron_sloys3 = np.copy(Neuron_sloys)

    Neuron_sloys3 = np.hstack(([len(test_case_inputs)], Neuron_sloys3))

    for i in range(len(Neuron_sloys3) - 1):
        for j in range(Neuron_sloys3[i + 1]):
            # dynamic_const = deltas[i][j] * nonlin_deriv(reactions[i + 1][j]) * learn_rate
            arg = 0
            for k in range(Neuron_sloys3[i] + 1):
                arg += reactions[i][k] * total_weights[i][k][j]

            dynamic_const = deltas[i][j] * learn_rate * deriv_relu(arg)

            for k in range(Neuron_sloys3[i] + 1):
                total_weights[i][k][j] -= reactions[i][k] * dynamic_const


x = np.array([2, 4, 5, 4])
inputs_number = len(x)

Neuron_sloys = np.array([2])
total_weight = construct(inputs_number, Neuron_sloys)
# total_weight номер строки - номер левого нейрона, номер столбца - номер правого нейрона

react = forward_prop(x, total_weight, inputs_number, Neuron_sloys)
# structure_network()
#
# x_train_data = np.array([[5.1, 3.5, 1.4, 0.2],
#                          [4.9, 3.1, 1.5, 0.1],
#                          [5.0, 3.5, 1.3, 0.3],
#                          [5.1, 3.5, 1.4, 0.3],
#                          [5.4, 3.7, 1.5, 0.2],
#                          [5.0, 3.4, 1.5, 0.2],
#                          [4.7, 3.2, 1.6, 0.2],
#                          [4.9, 3.6, 1.4, 0.1],
#                          [6.3, 3.3, 4.7, 1.6],
#                          [6.2, 2.2, 4.5, 1.5],
#                          [5.6, 2.5, 3.9, 1.1],
#                          [5.9, 3.0, 4.2, 1.5],
#                          [5.8, 2.6, 4.0, 1.2],
#                          [6.3, 2.5, 4.9, 1.5],
#                          [6.0, 2.7, 5.1, 1.6],
#                          [6.4, 2.9, 4.3, 1.3],
#                          [6.5, 3.0, 5.5, 1.8],
#                          [6.7, 3.3, 5.7, 2.1],
#                          [7.2, 3.2, 6.0, 1.8],
#                          [6.4, 2.8, 5.6, 2.1],
#                          [6.8, 3.2, 5.9, 2.3],
#                          [6.4, 2.7, 5.3, 1.9],
#                          [7.2, 3.2, 6.0, 1.8],
#                          [6.4, 2.8, 5.6, 2.2]])
#
# y_train_data = np.array([[0, 0, 0],
#                          [0, 0, 0],
#                          [0, 0, 0],
#                          [0, 0, 0],
#                          [0, 0, 0],
#                          [0, 0, 0],
#                          [0, 0, 0],
#                          [0, 0, 0],
#                          [1, 1, 1],
#                          [1, 1, 1],
#                          [1, 1, 1],
#                          [1, 1, 1],
#                          [1, 1, 1],
#                          [1, 1, 1],
#                          [1, 1, 1],
#                          [1, 1, 1],
#                          [2, 2, 2],
#                          [2, 2, 2],
#                          [2, 2, 2],
#                          [2, 2, 2],
#                          [2, 2, 2],
#                          [2, 2, 2],
#                          [2, 2, 2],
#                          [2, 2, 2]])
#
# for j in range(10):
#     total_weight = construct(inputs_number, Neuron_sloys)
#
#     for i in range(len(x_train_data)):
#         x, out = x_train_data[i], y_train_data[i]
#         backprop(total_weight, x, out, Neuron_sloys)
#
#     x1 = [5.6, 2.8, 4.9, 2.0]  # virginica
#     x2 = [5.0, 3.4, 1.6, 0.4]  # setosa
#     x3 = [5.5, 2.4, 3.8, 1.1]  # versicolor
#     y = ["setosa", "versicolor", "virginica"]
#     prediction1 = y[np.argmax(forward_prop(x1, total_weight, 4, Neuron_sloys)[-1])]
#     prediction2 = y[np.argmax(forward_prop(x2, total_weight, 4, Neuron_sloys)[-1])]
#     prediction3 = y[np.argmax(forward_prop(x3, total_weight, 4, Neuron_sloys)[-1])]
#     print(prediction1, prediction2, prediction3)  # virginica setosa versicolor
# print(total_weight)
t1 = time.time()

for i in range(100000):
    x, out = generator()
    backprop(total_weight, x, out, Neuron_sloys)

t2 = time.time()
print(t2 - t1)

for i in range(5):
    x, _ = generator()
    print("input: " + str(x))
    print("average: " + str(forward_prop(x, total_weight, 4, Neuron_sloys)[-1]))
    print()

# print(total_weight)
# structure_network()
