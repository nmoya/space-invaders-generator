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
        return None


def empty(size_x: int, size_y: int, default: int = 0):
    return [[default for _ in range(size_x)] for _ in range(size_y)]


def fill_around_point(array, x: int, y: int, value: int, depth: int):
    size_x = len(array[0])
    size_y = len(array)
    queue = Queue()
    queue.enqueue((y, x))
    visited = set()
    visited.add((y, x))

    while not queue.is_empty() and depth > 0:
        current_y, current_x = queue.dequeue()
        for ny, nx in adj_8(current_x, current_y, size_x, size_y):
            if (ny, nx) not in visited:
                array[ny][nx] = value
                visited.add((ny, nx))
                queue.enqueue((ny, nx))
        depth -= 1


def adj_8(x: int, y: int, size_x: int, size_y: int):
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < size_x and 0 <= ny < size_y:
                neighbors.append((ny, nx))
    return neighbors


def mirror(array):
    size_x = len(array[0])
    size_y = len(array)
    for y in range(size_y):
        for x in range(math.floor(size_x / 2)):
            array[y][size_x - x - 1] = array[y][x]


def fill_eyes_gap(array, eye_x: int, eye_y: int):
    pos_x = eye_x + 1
    pos_y = eye_y
    max_x = math.ceil(len(array[0]) / 2)
    for x in range(pos_x, max_x):
        array[pos_y][x] = 1


def randomize_body(array, eye_x: int, eye_y: int):
    head_counter = 4
    legs_counter = 8
    size_x = len(array[0])
    size_y = len(array)
    queue = Queue()
    for ny, nx in adj_8(eye_x, eye_y, size_x, size_y):
        queue.enqueue((ny, nx))
    visited = set()
    visited.add((eye_y, eye_x))

    while not queue.is_empty():
        current_y, current_x = queue.dequeue()
        visited.add((current_y, current_x))
        if current_x > math.ceil(size_x / 2) - 1:
            continue
        if array[current_y][current_x] == 0:
            if current_y <= eye_y and random.random() < (1 / head_counter):
                head_counter += 1
                array[current_y][current_x] = 1
            elif current_y > eye_y and random.random() < (1 / legs_counter):
                legs_counter += 1
                array[current_y][current_x] = 1
        for ny, nx in adj_8(current_x, current_y, size_x, size_y):
            if (ny, nx) not in visited and array[current_y][current_x] == 1:
                queue.enqueue((ny, nx))


class Invader:
    def __init__(self, size_x: int, size_y: int):
        self.size_x = size_x
        self.size_y = size_y
        self.left_eye = (0, 0)
        self.right_eye = (0, 0)
        self.body = empty(size_x, size_y)

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
        fill_around_point(self.body, self.left_eye[0], self.left_eye[1], 1, 1)
        mirror(self.body)
        fill_eyes_gap(self.body, self.left_eye[0], self.left_eye[1])

    def gen_body(self):
        randomize_body(self.body, self.left_eye[0], self.left_eye[1])
        mirror(self.body)

    def gen(self):
        self.gen_eyes()
        self.gen_body()


def draw_invader(invader: Invader, scale: int, body_color: tuple = (255, 255, 255), eye_color: tuple = (255, 0, 0)):
    image = Image.new(mode="RGB", size=(invader.size_x, invader.size_y), color=(0, 0, 0))
    for y, row in enumerate(invader.body):
        for x, value in enumerate(row):
            if (x, y) == invader.left_eye or (x, y) == invader.right_eye:
                image.putpixel((x, y), eye_color)
            if value == 1:
                image.putpixel((x, y), body_color)
    lg = image.resize((image.size[0] * scale, image.size[1] * scale), resample=Image.NEAREST)
    return lg


def save_invader(
    invader: Invader,
    filename: str,
    scale: int = 10,
    body_color: tuple = (255, 255, 255),
    eye_color: tuple = (255, 0, 0),
):
    image = draw_invader(invader, scale, body_color, eye_color)
    image.save(filename)


def gen_batch_invaders(count: int, size_x: int, size_y: int):
    invaders = []
    for i in tqdm(range(count)):
        invader = Invader(size_x, size_y)
        invader.gen()
        invaders.append(invader)
        save_invader(
            invader, f"./invaders/invader_{i}.png", scale=100, body_color=(255, 176, 0), eye_color=(225, 225, 225)
        )
    return invaders


if __name__ == "__main__":
    invader = Invader(13, 13)
    invader.gen()
    # save_invader(invader, "debug_large.png", scale=10, body_color=(255, 255, 255), eye_color=(255, 0, 0))

    gen_batch_invaders(50, 13, 13)
