import numpy as np
import matplotlib.pyplot as plt
import json


def relu(arg):
    return np.maximum(arg, 0)


def d_relu(arg):
    if arg <= 0:
        return np.random.normal(0.01, 0.05)
    return 1


def sigmoid(arg):
    return 1 / (1 + np.exp(-arg))


def d_sigmoid(arg):
    return arg * (1 - arg)


def generator():
    # x = np.random.randint(0, 100, size=4)
    x_array = np.abs(np.random.normal(0, 1, size=4))
    y_array = np.array([x_array[:2].mean(), x_array[2:].mean()])
    return x_array, y_array


def init_weights(n, m):
    """ инициализирует случайные веса в промежутке -0.5 - 0.5
    в матрице размера n,m  , где n - количество нейронов в предыдущем слое
    m - количество нейронов в следующем слое
    """
    matrix_weights = np.random.sample((n, m)) - 0.5
    return matrix_weights


def construct(number_inputs, neuron_layer):
    total_weights = [init_weights(number_inputs + 1, neuron_layer[0])]

    for i in range(len(neuron_layer) - 1):
        total_weights.append(init_weights(neuron_layer[i] + 1, neuron_layer[i + 1]))

    return total_weights


def forward_prop(x_inputs, total_weights, input_number, neuron_layer):
    x_inputs = np.append(x_inputs, 1)
    neuron_layer = np.append([input_number], neuron_layer)

    reactions = [np.array(x_inputs)]

    for i in range(len(neuron_layer) - 1):
        vector_reactions = np.zeros(neuron_layer[i + 1])

        for j in range(neuron_layer[i + 1]):
            for k in range(neuron_layer[i] + 1):
                vector_reactions[j] += reactions[i][k] * total_weights[i][k][j]

        vector_reactions = relu(vector_reactions)

        if i != len(neuron_layer) - 2:
            vector_reactions = np.append(vector_reactions, 1)

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
                    color_link = 'white'
                    if total_weight[i][j][k] < 0:
                        color_link = 'blue'
                    elif total_weight[i][j][k] > 0:
                        color_link = 'red'

                    plt.plot([i, i + 1], [len(react[i]) / 2 - j, len(react[i + 1]) / 2 - k],
                             color=color_link,
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


def backprop(total_weights, test_case_inputs, test_case_output, Neuron_sloys):
    Error = []

    for epoch in range(len(test_case_inputs)):

        input = test_case_inputs[epoch]
        output = test_case_output[epoch]

        learn_rate = 0.1 / ((epoch + 1) ** 0.3)

        reactions = forward_prop(input, total_weights, len(input), Neuron_sloys)

        Error.append(MSE_loss(output, reactions[-1]).mean())

        deltas = [np.array(MSE_loss2(output, reactions[-1]))]

        if len(Neuron_sloys) > 1:
            for i in range(len(Neuron_sloys) - 2, -1, -1):

                vector_deltas = np.zeros(Neuron_sloys[i])

                for j in range(Neuron_sloys[i]):
                    for k in range(Neuron_sloys[i + 1]):
                        vector_deltas[j] += deltas[0][k] * total_weights[i + 1][j][k]
                deltas.insert(0, np.array(vector_deltas))

        Neuron_sloys3 = np.copy(Neuron_sloys)

        Neuron_sloys3 = np.hstack(([len(input)], Neuron_sloys3))

        for i in range(len(Neuron_sloys3) - 1):

            for j in range(Neuron_sloys3[i + 1]):

                dynamic_const = deltas[i][j] * learn_rate * d_relu(reactions[i + 1][j])

                for k in range(Neuron_sloys3[i] + 1):
                    total_weights[i][k][j] -= reactions[i][k] * dynamic_const

    return Error


x = np.array([2, 4, 5, 4])
inputs_number = len(x)

Neuron_sloys = np.array([4, 2])
total_weight = construct(inputs_number, Neuron_sloys)
react = forward_prop(x, total_weight, inputs_number, Neuron_sloys)

# test_input = []
# test_output = []
#
# for i in range(10000):
#     x, out = generator()
#     test_input.append(x)
#     test_output.append(out)
#
# Er1 = backprop(total_weight, test_input, test_output, Neuron_sloys)
# backprop(total_weight, test_input, test_output, Neuron_sloys)

test_input = np.array([[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 4.7, 1.6], [6.5, 3.0, 5.5, 1.8],
                       [4.9, 3.1, 1.5, 0.1], [6.2, 2.2, 4.5, 1.5], [6.7, 3.3, 5.7, 2.1],
                       [5.0, 3.5, 1.3, 0.3], [5.6, 2.5, 3.9, 1.1], [7.2, 3.2, 6.0, 1.8],
                       [5.1, 3.5, 1.4, 0.3], [5.9, 3.0, 4.2, 1.5], [6.4, 2.8, 5.6, 2.1],
                       [5.4, 3.7, 1.5, 0.2], [5.8, 2.6, 4.0, 1.2], [6.8, 3.2, 5.9, 2.3],
                       [5.0, 3.4, 1.5, 0.2], [6.3, 2.5, 4.9, 1.5], [6.4, 2.7, 5.3, 1.9],
                       [4.7, 3.2, 1.6, 0.2], [6.0, 2.7, 5.1, 1.6], [7.2, 3.2, 6.0, 1.8],
                       [4.9, 3.6, 1.4, 0.1], [6.4, 2.9, 4.3, 1.3], [6.4, 2.8, 5.6, 2.2]])

test_output = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2],
                        [0, 0, 0], [1, 1, 1], [2, 2, 2],
                        [0, 0, 0], [1, 1, 1], [2, 2, 2],
                        [0, 0, 0], [1, 1, 1], [2, 2, 2],
                        [0, 0, 0], [1, 1, 1], [2, 2, 2],
                        [0, 0, 0], [1, 1, 1], [2, 2, 2],
                        [0, 0, 0], [1, 1, 1], [2, 2, 2],
                        [0, 0, 0], [1, 1, 1], [2, 2, 2]])

Neuron_sloys = np.array([4, 3])

for j in range(10):
    total_weight = construct(inputs_number, Neuron_sloys)
    backprop(total_weight, test_input, test_output, Neuron_sloys)

    x1 = [5.6, 2.8, 4.9, 2.0]  # virginica
    x2 = [5.0, 3.4, 1.6, 0.4]  # setosa
    x3 = [5.5, 2.4, 3.8, 1.1]  # versicolor
    y = ["setosa", "versicolor", "virginica"]

    prediction1 = y[np.argmax(forward_prop(x1, total_weight, 4, Neuron_sloys)[-1])]
    prediction2 = y[np.argmax(forward_prop(x2, total_weight, 4, Neuron_sloys)[-1])]
    prediction3 = y[np.argmax(forward_prop(x3, total_weight, 4, Neuron_sloys)[-1])]

    print(prediction1, prediction2, prediction3)  # virginica setosa versicolor

# np.save('w1', total_weight[0])
# np.save('w2', total_weight[1])

# for i in range(10):
#     x, _ = generator()
#     print("input: " + str(x))
#     print("average: " + str(forward_prop(x, total_weight, 4, Neuron_sloys)[-1]))
#     print()
#
# plt.figure()
# plt.grid()
# plt.plot(Er1)
#
# plt.show()
#
# def allan_deviation(z, dt, tau):
#     ADEV = np.zeros(tau.size, dtype='double')
#     n = z.size
#     maxi = 0
#     for i in range(tau.size):
#         if tau[i] * 3 < n:
#             maxi = i
#             sigma2 = np.sum((z[2 * tau[i]::1] - 2 * z[tau[i]:-tau[i]:1]
#                              + z[0:-2 * tau[i]:1]) ** 2)
#             ADEV[i] = np.sqrt(0.5 * sigma2 / (n - 2 * tau[i])) / tau[i] / dt
#         else:
#             break
#     return tau[:maxi].astype(np.double) * dt, ADEV[:maxi]
#
#
#
# tau = np.arange(1, 10)
# tau = np.append(tau, np.arange(10, 100, 10))
# tau = np.append(tau, np.arange(100, 1000, 100))
# tau = np.append(tau, np.arange(1000, 10000, 1000))
# tau = np.append(tau, np.arange(10000, 100000, 10000))
#
# al1 = np.array(allan_deviation(np.array(Er1), 1, tau))
#
# ###
# plt.figure(3)
# plt.loglog(al1[0], al1[1])
# plt.grid()
# plt.show()
