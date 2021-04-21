import numpy as np
import matplotlib.pyplot as plt
import time


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


def mse_loss(y_train, y_prediction):
    return (y_train - y_prediction) ** 2


def d_mse_loss(y_train, y_prediction):
    return - 2 * (y_train - y_prediction)


def show_structure(total_weights, network_structure, x_input):
    plt.figure()

    reaction = forward_prop(x_input, total_weights, network_structure, True)

    amount_layers = len(network_structure) - 1

    for i in range(amount_layers + 1):
        for j in range(len(reaction[i])):
            if i != amount_layers:

                # Связи
                scope = len(reaction[i + 1]) - 1
                if i == amount_layers - 1:
                    scope = len(reaction[i + 1])

                # Знак связи
                for k in range(scope):
                    color_link = 'white'
                    if total_weights[i][j][k] < 0:
                        color_link = 'blue'
                    elif total_weights[i][j][k] > 0:
                        color_link = 'red'

                    plt.plot([i, i + 1], [len(reaction[i]) / 2 - j, len(reaction[i + 1]) / 2 - k],
                             color=color_link,
                             alpha=abs(total_weights[i][j][k]) * 2,
                             linewidth=0.5)
            # Нейроны
            cc = i / (amount_layers + 1) * 0.8
            color = (1 - cc, 0, cc)
            plt.plot(i, len(reaction[i]) / 2 - j,
                     marker='o', markersize=23, markerfacecolor="white", markeredgewidth=1.4, color=color)

            plt.text(i, len(reaction[i]) / 2 - j, round(reaction[i][j - len(reaction[i])], 3),
                     horizontalalignment='center',
                     verticalalignment='center', fontsize=6.5, fontdict={'color': 'black'})


def generator(amount=1):
    # x_array = np.random.randint(0, 100, size=4)
    # x_array = np.random.random(size=4) - 0.5

    x_array = np.abs(np.random.normal(0, 1, (amount, 4)))
    y_array = np.array([[x[:2].mean(), x[2:].mean()] for x in x_array])

    return x_array, y_array


def init_weights(n, m):
    """ инициализирует случайные веса в промежутке -0.5 - 0.5
    в матрице размера n,m  , где n - количество нейронов в предыдущем слое
    m - количество нейронов в следующем слое
    """
    matrix_weights = np.random.sample((n, m)) - 0.5
    return matrix_weights


def construct(network_structure):
    total_weights = []

    for i in range(len(network_structure) - 1):
        total_weights.append(init_weights(network_structure[i] + 1, network_structure[i + 1]))
    return total_weights


def forward_prop(x_inputs, total_weights, network_structure, full=False):
    x_inputs = np.append(x_inputs, -1)

    reactions = [np.array(x_inputs)]

    for i in range(len(network_structure) - 1):
        vector_reactions = np.zeros(network_structure[i + 1])

        for j in range(network_structure[i + 1]):
            for k in range(network_structure[i] + 1):
                vector_reactions[j] += reactions[i][k] * total_weights[i][k][j]

        vector_reactions = relu(vector_reactions)

        if i != len(network_structure) - 2:
            vector_reactions = np.append(vector_reactions, -1)
        reactions.append(vector_reactions)

    if full:
        return reactions
    return reactions[-1]


def back_prop(total_weights, test_case_inputs, test_case_outputs, network_structure):
    Error = []
    neuron_layer = network_structure[1:]

    for epoch in range(len(test_case_inputs)):

        inputs = test_case_inputs[epoch]
        outputs = test_case_outputs[epoch]

        learn_rate = 0.1 / ((epoch + 1) ** 0.3)

        reactions = forward_prop(inputs, total_weights, network_structure, True)

        Error.append(mse_loss(outputs, reactions[-1]).mean())

        deltas = [np.array(d_mse_loss(outputs, reactions[-1]))]

        if len(neuron_layer) > 1:
            for i in range(len(neuron_layer) - 2, -1, -1):

                vector_deltas = np.zeros(neuron_layer[i])

                for j in range(neuron_layer[i]):
                    for k in range(neuron_layer[i + 1]):
                        vector_deltas[j] += deltas[0][k] * total_weights[i + 1][j][k]
                deltas.insert(0, np.array(vector_deltas))

        for i in range(len(network_structure) - 1):

            for j in range(network_structure[i + 1]):

                dynamic_const = deltas[i][j] * learn_rate * d_relu(reactions[i + 1][j])

                for k in range(network_structure[i] + 1):
                    total_weights[i][k][j] -= reactions[i][k] * dynamic_const

    return Error


x111 = np.array([100, 1, 600, 352])
inputs_number = len(x111)

Neuron_sloys = np.array([2])
Neuron_sloys1 = np.array([4, 2])
total_weight = construct(Neuron_sloys1)
react = forward_prop(x111, total_weight, Neuron_sloys1)

test_input, test_output = generator(1000)

show_structure(total_weight, Neuron_sloys1, x111)

print(forward_prop(x111, total_weight, Neuron_sloys1))
t1 = time.time()
Er1 = back_prop(total_weight, test_input, test_output, Neuron_sloys1)
t2 = time.time()
print(t1 - t2)
# total_weight = np.round(total_weight, 3)

show_structure(total_weight, Neuron_sloys1, x111)
print(forward_prop(x111, total_weight, Neuron_sloys1))
print(total_weight)
print(Neuron_sloys)
plt.show()
# for i in range(10):
#     x, _ = generator()
#     print("input: " + str(x))
#     print("average: " + str(forward_prop(x, total_weight, 4, Neuron_sloys1)))
#     print()

plt.figure()
plt.grid()
plt.plot(Er1)

plt.show()
