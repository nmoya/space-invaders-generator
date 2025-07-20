import copy
import math
import random

from PIL import Image
from tqdm import tqdm

invader = [
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1],
]


class Queue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        raise ValueError("Queue is empty")


class Matrix:
    def __init__(self, size_x: int, size_y: int):
        self.size_x = size_x
        self.size_y = size_y
        self.data = [[0 for _ in range(size_x)] for _ in range(size_y)]

    def __iter__(self):
        for y in range(self.size_y):
            for x in range(self.size_x):
                yield (x, y, self.data[y][x])

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self.size_x != other.size_x or self.size_y != other.size_y:
            return False
        for y in range(self.size_y):
            for x in range(self.size_x):
                if self.data[y][x] != other.data[y][x]:
                    return False
        return True


class InvaderBody:
    def __init__(self, invader, size_x: int, size_y: int):
        self.invader = invader
        self.size_x = size_x
        self.size_y = size_y
        self.half_x = math.ceil(size_x / 2)
        self.matrix = Matrix(size_x, size_y)
        self.invader.save_body_iteration(self.matrix)

    def set_cell(self, x: int, y: int, value: int):
        if not (0 <= x < self.size_x and 0 <= y < self.size_y):
            raise IndexError("Coordinates out of bounds")

        self.matrix.data[y][x] = value
        self.invader.save_body_iteration(self.matrix)

    def get_cell(self, x: int, y: int) -> int:
        if not (0 <= x < self.size_x and 0 <= y < self.size_y):
            raise IndexError("Coordinates out of bounds")
        return self.matrix.data[y][x]

    def adj_8(self, x: int, y: int):
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size_x and 0 <= ny < self.size_y:
                    neighbors.append((ny, nx))
        return neighbors

    def mirror(self):
        for y in range(self.size_y):
            for x in range(self.half_x):
                self.set_cell(self.size_x - x - 1, y, self.get_cell(x, y))

    def fill_around_point(self, x: int, y: int, value: int, depth: int):
        queue = Queue()
        queue.enqueue((y, x))
        visited = set()
        visited.add((y, x))

        while not queue.is_empty() and depth > 0:
            current_y, current_x = queue.dequeue()
            for ny, nx in self.adj_8(current_x, current_y):
                if (ny, nx) not in visited:
                    self.set_cell(nx, ny, value)
                    visited.add((ny, nx))
                    queue.enqueue((ny, nx))
            depth -= 1

    def fill_eyes_gap(self, eye_x: int, eye_y: int):
        pos_x = eye_x + 1
        pos_y = eye_y
        for x in range(pos_x, self.half_x):
            self.set_cell(x, pos_y, 1)

    def randomize(self, eye_x: int, eye_y: int):
        max_dist = max(self.half_x, self.size_y)
        falloff = 0.75
        visited = set()
        queue = Queue()
        for ny, nx in self.adj_8(eye_x, eye_y):
            visited.add((ny, nx))
            queue.enqueue((ny, nx))
        visited.add((eye_y, eye_x))

        while not queue.is_empty():
            current_y, current_x = queue.dequeue()
            if current_x > self.half_x - 1:
                continue
            if self.get_cell(current_x, current_y) == 0:
                distance = abs(current_y - eye_y) + abs(current_x - eye_x)
                prob = 1.0 - (distance / max_dist) ** falloff
                if random.random() < prob:
                    self.set_cell(current_x, current_y, 1)
            for ny, nx in self.adj_8(current_x, current_y):
                if (ny, nx) not in visited and self.get_cell(current_x, current_y) == 1:
                    visited.add((ny, nx))
                    queue.enqueue((ny, nx))


class Invader:
    def __init__(self, size_x: int, size_y: int):
        self.size_x = size_x
        self.size_y = size_y
        self.body_iterations = []
        self.left_eye = (0, 0)
        self.right_eye = (0, 0)
        self.body = InvaderBody(self, size_x, size_y)

    def save_body_iteration(self, array: Matrix):
        if len(self.body_iterations) > 0 and array == self.body_iterations[-1]:
            return
        deep_copy = copy.deepcopy(array)
        self.body_iterations.append(deep_copy)

    def set_eyes(self):
        min_limit_x = math.floor(self.size_x / 4)
        max_limit_x = math.ceil(self.size_x / 2) - 2
        min_limit_y = math.floor(self.size_y / 3)
        max_limit_y = math.floor(self.size_y / 3) * 2

        eye_position = (random.randint(min_limit_x, max_limit_x), random.randint(min_limit_y, max_limit_y))
        self.left_eye = eye_position
        self.right_eye = (self.size_x - eye_position[0] - 1, eye_position[1])

    def gen_eyes(self):
        self.set_eyes()
        self.body.fill_around_point(self.left_eye[0], self.left_eye[1], 1, 1)
        self.body.mirror()
        self.body.fill_eyes_gap(self.left_eye[0], self.left_eye[1])

    def gen_body(self):
        self.body.randomize(self.left_eye[0], self.left_eye[1])
        self.body.mirror()

    def gen(self):
        self.gen_eyes()
        self.gen_body()

    def render(self, matrix: Matrix, scale: int, body_color: tuple = (255, 255, 255), eye_color: tuple = (255, 0, 0)):
        image = Image.new(mode="RGB", size=(self.size_x, self.size_y), color=(0, 0, 0))
        for x, y, value in matrix:
            if (x, y) == self.left_eye or (x, y) == self.right_eye:
                image.putpixel((x, y), eye_color)
            if value == 1:
                image.putpixel((x, y), body_color)
        lg = image.resize((image.size[0] * scale, image.size[1] * scale), resample=Image.NEAREST)
        return lg

    def save_png(
        self,
        filename: str,
        scale: int = 10,
        body_color: tuple = (255, 255, 255),
        eye_color: tuple = (255, 0, 0),
    ):
        image = self.render(self.body.matrix, scale, body_color, eye_color)
        image.save(filename)

    def save_gif(
        self,
        filename: str,
        scale: int = 10,
        body_color: tuple = (255, 255, 255),
        eye_color: tuple = (255, 0, 0),
    ):
        gif_loop_duration_seconds = 3
        images = [self.render(matrix, scale, body_color, eye_color) for matrix in self.body_iterations]
        images = images + ([images[-1]] * 10)
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            loop=0,
            duration=max(gif_loop_duration_seconds / len(images) * 1000, 1 / 48 * 1000),
        )


def gen_batch_invaders(count: int, size_x: int, size_y: int):
    for i in tqdm(range(count)):
        invader = Invader(size_x, size_y)
        invader.gen()
        invader.save_png(f"./invaders/invader_{i}.png", scale=100, body_color=(255, 176, 0), eye_color=(225, 225, 225))
        invader.save_gif(f"./invaders/invader_{i}.gif", scale=100, body_color=(255, 176, 0), eye_color=(225, 225, 225))


if __name__ == "__main__":
    # invader = Invader(7, 7)
    # invader.gen()
    # invader.save("debug.png", scale=100, body_color=(255, 255, 255), eye_color=(255, 0, 0))
    # save_invader(invader, "debug_large.png", scale=10, body_color=(255, 255, 255), eye_color=(255, 0, 0))

    gen_batch_invaders(50, 12, 12)
