from tkinter import *
import numpy as np

size = 5  # Размер квадратного поля клеточного автомата size * size
px = 20  # Размер пикселя на графике

N = 4  # Параметры условий прятания и активизации клеток
M = 3

"""
Случайное начальное распределение активных и спрятанных клеток,
если D[i][i] = 1 если D[i][j] = 1 - клетка активна, если D[i][j] = 0 - клетка спрятана.
Зануление граничных элементов нужно для создания рамки на рисунке
(size + 2 * size + 2)  нужно для рамки на рисунке
"""
condition_zero = np.random.randint(0, 2, size=(size + 2, size + 2))
condition_zero[0] = 0
condition_zero[-1] = 0
condition_zero[:, 0] = 0
condition_zero[:, -1] = 0


# Функция, которая считает активных соседей. Так как занчения либо ноль либо единица, то просто суммируем 8 значений,
# находящихся рядом с нужной клеткой
def neighbors(condition, coord_x, coord_y):
    n = 0
    for i in [coord_x - 1, coord_x, coord_x + 1]:
        n += condition[coord_y - 1][i]
        n += condition[coord_y][i]
        n += condition[coord_y + 1][i]
    n -= condition[coord_y][coord_x]
    return n


# Функция, которая считает следующее состояние автомата по предыдущему
def condition_next(condition_prev):
    D_next = []  # Создание новой матрицы, заполненной нулями
    [D_next.append(np.zeros(size + 2)) for i in range(size + 2)]
    D_next = np.array(D_next)

    for i in range(1, size + 1):  # Двойной цикл, для прохождения по каждому
        for j in range(1, size + 1):  # элементу матрицы
            if condition_prev[j][i] == 1:  # Если клетка была активна, и если количество активных соседей
                if neighbors(condition_prev, i, j) > N:  # больше заданного числа N, то клетка прячется (D[i][j] = 0).
                    D_next[j][i] = 0  # Иначе, становится активной (D[i][j] = 1)
                else:
                    D_next[j][i] = 1
            else:
                if neighbors(condition_prev, i, j) <= M:  # Если клетка была не активной, и если количество активных
                    D_next[j][i] = 1  # соседей не больше заданного числа M, то клетка активизируется
                else:  # (D[i][j] = 1). Иначе, прячется (D[i][j] = 0).
                    D_next[j][i] = 0
    return D_next


# Функция, которая обновляет картинку. На вход подается текущее состояние автомата, в зависимости от которого клетки
# закрашиваются в черный или в зеленый цвет. Делает синюю рамку вокруг всего поля.
def pic(condition):
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            if condition[j][i] == 1:
                canvas.create_rectangle(i * px, j * px, (i + 1) * px, (j + 1) * px, fill="green")
            else:
                canvas.create_rectangle(i * px, j * px, (i + 1) * px, (j + 1) * px, fill="black")
    for i in range(0, size + 2):
        canvas.create_rectangle(i * px, 0, (i + 1) * px, px, fill="blue")
        canvas.create_rectangle(i * px, (size + 1) * px, (i + 1) * px, (size + 2) * px, fill="blue")
        canvas.create_rectangle(0, (i + 1) * px, px, (size + 1) * px, fill="blue")
        canvas.create_rectangle((size + 1) * px, (i + 1) * px, (size + 2) * px, (size + 1) * px, fill="blue")


# Создание графического интерфейса
window = Tk()
window.title('Дюны')
canvas = Canvas(window, width=(size + 2) * px, heigh=(size + 2) * px, bg='black')
pic(condition_zero)


# Функция для обновления картинки при нажатии на кнопку. Вызывает сначала функцию счета следующего состояния автомата,
# после чего строит новую картинку
def click():
    global condition_zero
    D_new = condition_next(condition_zero)
    pic(D_new)
    condition_zero = D_new
    return D_new


Next = Button(text='Next', command=click)  # Кнопка для перехода к следующей итерации

Next.pack()
canvas.pack()
window.mainloop()
