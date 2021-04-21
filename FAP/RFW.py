import numpy as np
from matplotlib import pyplot as plt

NN = 4              #Количество элементов

h = 0.01            #Шаг интегрирования
T = 0.01          #Постоянная времени
N = 10000           #Количество итераций

kk = []             #Коэффициенты связи
for i in range(NN):
    kk.append(4*np.pi)

# kk[0] = 0


w01 = 5*10^9            #Начальные частоты генераторов
w02 = 5*10^9

ws = (kk[0] * w01 + kk[-1] * w02) / (kk[0] + kk[-1])    #Частота генерации

delw1 = w01 - ws
delw2 = w02 - ws

nn1 = np.random.normal(0, 0.1, N)    # БГШ1
nn2 = np.random.normal(0, 0.1, N)  # БГШ1

n11 = np.random.normal(0, 1, N)    # БГШ1
n21 = np.random.normal(0, 1, N)
n12 = np.random.normal(0, 1, N)    # БГШ1
n22 = np.random.normal(0, 1, N)

### РК4
def rk4(fun, xyz, n):

    zz = [xyz]

    for i in range(1, n):
        k1 = fun(zz[-1], i)
        k2 = fun(zz[-1] + h * k1 / 2, i)
        k3 = fun(zz[-1] + h * k2 / 2, i)
        k4 = fun(zz[-1] + h * k3, i)

        zz1 = zz[-1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        zz.append(zz1)

    return zz


### Уравнения
k1 = 1
k2 = 1
def fun2(y, i):
    dy = np.zeros(6)
    c = 1

    dy[0] = y[1] + nn1[i] + y[2]
    dy[1] = c*(delw1 - y[1] - k1 * np.sin(y[3] - y[0]) + 0.1*n12[i]) / T
    dy[2] = 0.5*n11[i]

    dy[3] = y[4] + nn2[i] + y[5]
    dy[4] = c*(delw2 - y[4] - k2 * np.sin(y[0] - y[3]) + 0.1*n22[i]) / T
    dy[5] = 0.5*n21[i]

    return dy

#НУ
y0 = []
for i in range(2):
    y0.append(1)
    y0.append(w01)
    y0.append(0)

A = np.array(rk4(fun2, y0, N))

k1 = 0

A1 = np.array(rk4(fun2, y0, N))

#ФП
plt.figure(1)
plt.grid()
for i in range(2):
    plt.plot(A[:, 3*i])
# plt.savefig('k0=0, k1=0, k2=0')
plt.show()


# Девиация Аллана
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
plt.figure(2)
plt.grid()

al = np.array(allan_deviation(A[1000:-1, 0], 0.01, tau))
plt.loglog(al[0], al[1], 'g', label = "Генератор в ансамбле")

al1 = np.array(allan_deviation(A1[1000:-1, 0], 0.01, tau))
plt.loglog(al1[0], al1[1], 'b', label = "Свободный генератор")

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\sigma$')
plt.legend()

#tau = np.append(tau, np.arange(10000, 100000, 10000))
#tau = np.append(tau, np.arange(100000, 1000000, 100000))
#tau = np.append(tau, np.arange(1000000, 10000000, 1000000))

# plt.savefig('СБЧ')
plt.show()