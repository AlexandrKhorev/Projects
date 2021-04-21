import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(1)
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
    return np.average((y_train - y_prediction) ** 2, 0)


def d_mse_loss(y_train, y_prediction):
    return - 2 * np.average(y_train - y_prediction, 0)


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


def generator_abs(amount=1):
    # x_array = np.random.randint(0, 100, size=4)
    # x_array = np.random.random(size=4) - 0.5

    x_array = np.abs(np.random.normal(0, 1, (amount, 4)))
    y_array = np.array([np.array([x[:2].mean(), x[2:].mean()]) for x in x_array])

    return x_array, y_array


def generator_sign(amount=1):
    x_data = np.zeros((amount, 8))
    y_data = np.zeros((amount, 4))

    for i in range(len(x_data)):
        for j in range(len(x_data[i]) - 4):
            x_data[i][j] = np.random.normal(0, 1)
            if x_data[i][j] < 0:
                x_data[i][j + 4] = 0.2
            else:
                x_data[i][j + 4] = 0.8

    for i in range(len(y_data)):
        for j in range(len(y_data[i]) - 2):
            y_data[i][j] = x_data[i][j * 2:(j + 1) * 2].mean()
            if y_data[i][j] < 0:
                y_data[i][j + 2] = 0.2
            else:
                y_data[i][j + 2] = 0.8

    return np.abs(x_data), np.abs(y_data)


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
        vector_reactions = relu(np.dot(reactions[i], total_weights[i]))

        if i != len(network_structure) - 2:
            vector_reactions = np.append(vector_reactions, -1)

        reactions.append(vector_reactions)

    if full:
        return reactions
    return reactions[-1]


def back_prop(total_weights, test_case_inputs, test_case_outputs, network_structure):
    Loss_average = []
    length_batch = 32
    for epoch in range(len(test_case_inputs) // length_batch):

        learn_rate = 0.1 / ((epoch + 1) ** 0.3)
        alpha = 0.8

        inputs = test_case_inputs[epoch * length_batch:(epoch + 1) * length_batch]
        outputs = test_case_outputs[epoch * length_batch:(epoch + 1) * length_batch]

        prediction = np.array([forward_prop(x, total_weights, network_structure) for x in inputs])
        Loss_average.append(mse_loss(outputs, prediction).mean())
        deltas = [np.array(d_mse_loss(outputs, prediction))]

        reactions = np.array(
            forward_prop(inputs[np.random.randint(0, length_batch)], total_weights, network_structure, True))

        for i in reversed(range(len(network_structure) - 2)):
            vector_deltas = np.dot(total_weights[i + 1], deltas[0])
            deltas.insert(0, vector_deltas[:-1])

        inertia = [np.zeros_like(x) for x in total_weight]
        for i in range(len(network_structure) - 1):

            if i != len(network_structure) - 2:
                dr = np.array([d_relu(x) for x in reactions[i + 1][:-1]])
            else:
                dr = np.array([d_relu(x) for x in reactions[i + 1]])

            # gradient = np.outer(reactions[i], deltas[i]) * learn_rate * np.array([d_relu(x) for x in reactions[i + 1][:-1]])

            gradient = np.outer(reactions[i], deltas[i]) * learn_rate * dr
            total_weights[i] -= gradient - inertia[i]
            inertia[i] += alpha * gradient

    return Loss_average


# Neuron_Structure = np.array([8, 4])
Neuron_Structure = np.array([4, 2])

total_weight = construct(Neuron_Structure)

# test_input, test_output = generator_sign(10000)
test_input, test_output = generator_abs(10000)

Er1 = back_prop(total_weight, test_input, test_output, Neuron_Structure)

# x111 = np.array([100, 500, 600, 1000, 0.2, 0.8, 0.2, 0.2])
x111 = np.array([100, 500, 600, 1000])

# show_structure(total_weight, Neuron_Structure, x111)
print(forward_prop(x111, total_weight, Neuron_Structure))

show_structure(total_weight, Neuron_Structure, x111)
plt.figure()
plt.grid()
plt.plot(Er1)
plt.show()
