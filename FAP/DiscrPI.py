import numpy as np
from matplotlib import pyplot as plt
from time import time

N = 100000  # Количество итераций
NN = 2  # Количество элементов
delta_t = 10
dn = 10

ku = 0.01
kp = 0.05

n1 = np.random.normal(0, 8e-13, size=(NN, N))  # Частотный шум белый
n2 = np.random.normal(0, 1.5e-17, size=(NN, N))  # Частотный шум СБ
n3 = np.random.normal(0, 5e-13, size=(NN, N))  # Шум измерений белый


# n1 = np.random.normal(0, 0.5, size=(NN, N))  # Частотный шум белый
# n2 = np.random.normal(0, 0.01, size=(NN, N))  # Частотный шум СБ
# n3 = np.random.normal(0, 0.1, size=(NN, N))  # Шум измерений белый


# Сумма шумов = 0
# n3[1] = -n3[0]
# n3[1] = n3[0]


# y = [[fi1, w1, fi2, w2],[fi'1, w'1, fi'2, w'2], [fi''1, w''1, fi''2, w''2]]

def solve(bool_link):
    y = np.zeros((N, NN * 2))
    y2 = np.zeros((N, NN * 2))
    y[0] = np.array([0, 1e-13, 0, -1e-12])
    # y[0] = np.array([0, 10, 0, -10])
    y[1] = y[0]
    y[2] = y[1]

    # delta_phase = np.zeros(N)
    # delta_phase_prev = np.zeros(N)
    # delta_phase_prev_prev = np.zeros(N)
    for k in range(2, N - 1):

        for r in range(0, NN):
            # delta_phase[k] = y[k][r * 2] - y[k][(r - 1) * 2] + n3[r][k]
            # delta_phase_prev[k] = y[k - 1][r * 2] - y[k - 1][(r - 1) * 2] + n3[r][k - 1]
            # delta_phase_prev_prev[k] = y[k - 2][r * 2] - y[k - 2][(r - 1) * 2] + n3[r][k - 2]
            #
            # if k > dn + 10:
            #     delta_phase[k] = np.mean(delta_phase[k - dn + 1:k + 1])
            #     delta_phase_prev[k] = np.mean(delta_phase_prev[k - dn + 1:k + 1])
            #     delta_phase_prev_prev[k] = np.mean(delta_phase_prev[k - dn + 1:k + 1])
            #
            # delta_freq = (delta_phase[k] - delta_phase_prev[k]) / delta_t
            # delta_freq_prev = (delta_phase_prev[k] - delta_phase_prev_prev[k]) / delta_t

            delta_phase = y[k][r * 2] - y[k][(r - 1) * 2] + n3[r][k]
            delta_phase_prev = y[k - 1][r * 2] - y[k - 1][(r - 1) * 2] + n3[r][k - 1]
            delta_phase_prev_prev = y[k - 2][r * 2] - y[k - 2][(r - 1) * 2] + n3[r][k - 2]

            delta_freq = (delta_phase - delta_phase_prev) / delta_t
            delta_freq_prev = (delta_phase_prev - delta_phase_prev_prev) / delta_t

            # control = (- gx * delta_phase - gy * delta_freq) * bool_link
            control = (- ku * delta_freq - kp * (delta_freq - delta_freq_prev)) * bool_link

            y[k + 1][r * 2] = y[k][r * 2] + y[k][r * 2 + 1] * delta_t + control * delta_t + n1[r][k]
            y[k + 1][r * 2 + 1] = y[k][r * 2 + 1] + control + n2[r][k]

    return y


# solve(наличие связи)
A1 = solve(True)
A2 = solve(False)

if (0):
    plt.figure(1)
    plt.title("Phase")
    plt.grid()
    plt.xlabel("n")
    plt.ylabel(r'$\varphi$')
    for i in range(0, NN):
        plt.plot(A1[:, i * 2])
    # plt.savefig('4b1')

    plt.figure(4)
    plt.title("Freq")
    plt.grid()
    plt.xlabel("n")
    plt.ylabel(r'$\omega$')
    for i in range(0, NN):
        plt.plot(A1[:, i * 2 + 1])


# plt.savefig('4b2')


###
def allan_deviation(z, dt, tau):
    ADEV = np.zeros(tau.size, dtype='double')
    n = z.size
    maxi = 0
    for i in range(tau.size):
        if tau[i] * 3 < n:
            maxi = i
            sigma2 = np.sum((z[2 * tau[i]::1] - 2 * z[tau[i]:-tau[i]:1]
                             + z[0:-2 * tau[i]:1]) ** 2)
            ADEV[i] = np.sqrt(0.5 * sigma2 / (n - 2 * tau[i])) / tau[i] / dt
        else:
            break
    return tau[:maxi].astype(np.double) * dt, ADEV[:maxi]


###
tau = np.arange(1, 10)
tau = np.append(tau, np.arange(10, 100, 10))
tau = np.append(tau, np.arange(100, 1000, 100))
tau = np.append(tau, np.arange(1000, 10000, 1000))
tau = np.append(tau, np.arange(10000, 100000, 10000))

al1 = np.array(allan_deviation(A1[1000:-1, 0], delta_t, tau))
al2 = np.array(allan_deviation(A2[1000:-1, 0], delta_t, tau))

###
plt.figure(3)
plt.title("Allan Deviation")
plt.loglog(al1[0], al1[1], 'b', label="Связанный генератор")
plt.loglog(al2[0], al2[1], 'g', label="Свободный генератор")
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\sigma$')
plt.grid()
plt.legend()
# plt.savefig('4b3')

plt.show()
