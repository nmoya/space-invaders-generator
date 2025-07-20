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


class InvaderBody:
    def __init__(self, size_x: int, size_y: int):
        self.size_x = size_x
        self.size_y = size_y
        self.half_x = math.ceil(size_x / 2)
        self.body = [[0 for _ in range(size_x)] for _ in range(size_y)]

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
        size_x = len(self.body[0])
        size_y = len(self.body)
        for y in range(size_y):
            for x in range(self.half_x):
                self.body[y][size_x - x - 1] = self.body[y][x]

    def fill_around_point(self, x: int, y: int, value: int, depth: int):
        queue = Queue()
        queue.enqueue((y, x))
        visited = set()
        visited.add((y, x))

        while not queue.is_empty() and depth > 0:
            current_y, current_x = queue.dequeue()
            for ny, nx in self.adj_8(current_x, current_y):
                if (ny, nx) not in visited:
                    self.body[ny][nx] = value
                    visited.add((ny, nx))
                    queue.enqueue((ny, nx))
            depth -= 1

    def fill_eyes_gap(self, eye_x: int, eye_y: int):
        pos_x = eye_x + 1
        pos_y = eye_y
        for x in range(pos_x, self.half_x):
            self.body[pos_y][x] = 1

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
            if self.body[current_y][current_x] == 0:
                distance = abs(current_y - eye_y) + abs(current_x - eye_x)
                prob = 1.0 - (distance / max_dist) ** falloff
                if random.random() < prob:
                    self.body[current_y][current_x] = 1
            for ny, nx in self.adj_8(current_x, current_y):
                if (ny, nx) not in visited and self.body[current_y][current_x] == 1:
                    visited.add((ny, nx))
                    queue.enqueue((ny, nx))


class Invader:
    def __init__(self, size_x: int, size_y: int):
        self.size_x = size_x
        self.size_y = size_y
        self.left_eye = (0, 0)
        self.right_eye = (0, 0)
        self.body = InvaderBody(size_x, size_y)

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

    def draw_invader(self, scale: int, body_color: tuple = (255, 255, 255), eye_color: tuple = (255, 0, 0)):
        image = Image.new(mode="RGB", size=(self.size_x, self.size_y), color=(0, 0, 0))
        for y, row in enumerate(self.body.body):
            for x, value in enumerate(row):
                if (x, y) == self.left_eye or (x, y) == self.right_eye:
                    image.putpixel((x, y), eye_color)
                if value == 1:
                    image.putpixel((x, y), body_color)
        lg = image.resize((image.size[0] * scale, image.size[1] * scale), resample=Image.NEAREST)
        return lg

    def save(
        self,
        filename: str,
        scale: int = 10,
        body_color: tuple = (255, 255, 255),
        eye_color: tuple = (255, 0, 0),
    ):
        image = self.draw_invader(scale, body_color, eye_color)
        image.save(filename)


def gen_batch_invaders(count: int, size_x: int, size_y: int):
    for i in tqdm(range(count)):
        invader = Invader(size_x, size_y)
        invader.gen()
        invader.save(f"./invaders/invader_{i}.png", scale=100, body_color=(255, 176, 0), eye_color=(225, 225, 225))


if __name__ == "__main__":
    # invader = Invader(7, 7)
    # invader.gen()
    # invader.save("debug.png", scale=100, body_color=(255, 255, 255), eye_color=(255, 0, 0))
    # save_invader(invader, "debug_large.png", scale=10, body_color=(255, 255, 255), eye_color=(255, 0, 0))

    gen_batch_invaders(50, 7, 7)
