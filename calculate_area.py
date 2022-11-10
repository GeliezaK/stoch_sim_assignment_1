import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image, ImageDraw

def approx_area(n_samples, n_mandelbrot_samples, circle_r):
    total_area = np.pi*circle_r*circle_r
    frac = n_mandelbrot_samples/n_samples
    mandelbrot_area = total_area*frac
    return mandelbrot_area


def make_random_point(circle_r):
    # center of the circle (x, y)
    circle_x = 0
    circle_y = 0

    # random angle
    alpha = 2 * math.pi * np.random.random()
    # random radius
    r = circle_r * math.sqrt(np.random.random())
    # calculating coordinates
    c = complex((r * math.cos(alpha) + circle_x), (r * math.sin(alpha) + circle_y ))
    return c

def is_stable(c, num_iterations):
    # Check if the given complex number is part of the mandelbrot set
    z = 0
    for _ in range(num_iterations):
        z = z ** 2 + c
        if abs(z) > 2:
            return False
    return True

def draw_random_numbers(s, circle_r, i):
    # samples = np.zeros(s, dtype=np.csingle)
    samples = []
    for j in range(s):
        if j % 100000 == 0:
            print(f"sample {j}")

        c = make_random_point(circle_r)
        if is_stable(c, i):
            samples.append(c)

    samples = np.array(samples)
    return samples

if __name__ == '__main__':
    width = 1000
    height = 1000

    n_samples = int(1e6)
    n_iterations = 500

    circle_r = 2
    
    im = Image.new("RGB", (width, height), color ="white")
    draw = ImageDraw.Draw(im)
    samples = draw_random_numbers(n_samples, circle_r, n_iterations)
    n_mandel_samples = len(samples)

    area = approx_area(n_samples, n_mandel_samples, circle_r)
    print(f"Aproximated area of Mandelbrot set = {area}\n(Calculated using {n_samples} sample points and {n_iterations} iterations)")

    for i in range(n_mandel_samples):
        real = width/2 + np.real(samples[i]) * width/4
        imag = height/2 + np.imag(samples[i]) * height/4  # should be height/(2*radius)
        draw.point((real, imag), fill="black")
    im.show()