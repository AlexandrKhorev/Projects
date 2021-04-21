from tkinter import *
import random


class Segment(object):
    def __init__(self, x, y):
        self.instance = c.create_rectangle(x, y, x + PX, y + PX, fill='white')


class Snake(object):
    def __init__(self):
        self.segments = [Segment(PX, PX),
                         Segment(2 * PX, PX),
                         Segment(3 * PX, PX)]
        self.possible_motions = {"Down": (0, 1), "Up": (0, -1), "Left": (-1, 0), "Right": (1, 0)}
        self.motion = self.possible_motions["Right"]

    def move(self):
        """Движение змейки в указанном направлении"""

        for index in range(len(self.segments) - 1):
            segment_prev = self.segments[index].instance
            segment_next = self.segments[index + 1].instance
            c.coords(segment_prev, c.coords(segment_next))

        coord_head_prev = c.coords(self.segments[-2].instance)

        c.coords(self.segments[-1].instance,
                 coord_head_prev[0] + self.motion[0] * PX,
                 coord_head_prev[1] + self.motion[1] * PX,
                 coord_head_prev[2] + self.motion[0] * PX,
                 coord_head_prev[3] + self.motion[1] * PX)

    def change_direction(self, event):
        """Изменение направления движения"""
        if event.keysym in self.possible_motions:
            self.motion = self.possible_motions[event.keysym]

    def add_segment(self):
        _, _, x, y = c.coords(self.segments[0].instance)
        self.segments.insert(0, Segment(x - PX, y - PX))

    def reset_snake(self):
        for segment in self.segments:
            c.delete(segment.instance)


def create_apple():
    global Apple
    pos_x = random.randint(1, (SIZE - 1)) * PX
    pos_y = random.randint(1, (SIZE - 1)) * PX
    Apple = c.create_oval(pos_x, pos_y, pos_x + PX, pos_y + PX, fill='red')


def main():
    global In_game
    if In_game:
        snake.move()

        coord_head = c.coords(snake.segments[-1].instance)
        for coord in coord_head:
            if coord in [0, SIZE * PX]:
                In_game = False

        if coord_head == c.coords(Apple):
            snake.add_segment()
            c.delete(Apple)
            create_apple()

        for segment in snake.segments[:-1]:
            if coord_head == c.coords(segment.instance):
                In_game = False

        tk.after(100, main)
    else:
        lose()


def lose():
    c.itemconfig(game_over_text, state='normal')
    c.itemconfig(restart_text, state='normal')
    c.tag_bind(restart_text, "<Button-1>", restart)


def restart(event):
    global In_game
    snake.reset_snake()
    In_game = True
    c.delete(Apple)
    c.itemconfig(game_over_text, state='hidden')
    c.itemconfig(restart_text, state='hidden')
    start_game()


def start_game():
    global snake
    snake = Snake()
    create_apple()
    c.bind("<KeyPress>", snake.change_direction)
    main()


tk = Tk()
tk.title('Snake')
tk.resizable(0, 0)

In_game = True
PX = 20
SIZE = 20

c = Canvas(tk, width=SIZE * PX, height=SIZE * PX, bg='green', highlightthickness=0)
c.grid()
game_over_text = c.create_text(SIZE * PX / 2, SIZE * PX / 2, text="GAME OVER!", state='hidden')
restart_text = c.create_text(SIZE * PX / 2, SIZE * (PX + 10) / 2, text="Restart", state='hidden')
c.focus_set()

start_game()
tk.mainloop()
