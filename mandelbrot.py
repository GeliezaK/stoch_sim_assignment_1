import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import log
from PIL import Image
from dataclasses import dataclass

"""Code from https://realpython.com/mandelbrot-set-python/ - Color gradient"""

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def stability(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.escape_count(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def escape_count(self, c: complex, smooth=False) -> float:
        z = 0
        for iteration in range(self.max_iterations):
            z = z ** 2 + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return float(iteration + 1 - log(log(abs(z))) / log(2))
                return float(iteration)
        return float(self.max_iterations)

def denormalize(palette):
    return [
        tuple(int(channel * 255) for channel in color)
        for color in palette
    ]

def make_gradient(colors, interpolation="linear"):
    X = [i / (len(colors) - 1) for i in range(len(colors))]
    Y = [[color[i] for color in colors] for i in range(3)]
    channels = [interp1d(X, y, kind=interpolation) for y in Y]
    return lambda x: [np.clip(channel(x), 0, 1) for channel in channels]

if __name__ == '__main__':
    black = (0, 0, 0)
    blue = (0, 0, 1)
    orange = (0.5, 0.5, 0)
    navy = (0, 0, 0.5)
    yellow = (0,1,0)

    colors = [black, navy, blue, orange, yellow, black]
    gradient = make_gradient(colors, interpolation="cubic")
    num_colors = 256
    palette = denormalize([
        gradient(i / num_colors) for i in range(num_colors)
    ])
    print(palette[127])
    mandelbrot_set = MandelbrotSet(max_iterations=20, escape_radius=1000)

    width, height = 600, 400
    scale = 0.008

    image = Image.new(mode="RGB", size=(width, height))
    for y in range(height):
        for x in range(width):
            c = scale * complex(x - width / 2, height / 2 - y)
            stability = mandelbrot_set.stability(c, smooth=True)
            index = int(min(stability * len(palette), len(palette) - 1))
            color = palette[index % len(palette)]
            image.putpixel((x, y), color)
    image.show()