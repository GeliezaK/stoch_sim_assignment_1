import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image, ImageDraw

def approx_area(s):
    # fill two arrays uniformly distributed random values between -2 and 2
    in_set = 0
    samples = np.zeros((2,s))
    # for all random values in range, calculate if its in

    # Area is the fraction of points within mandelbrot divided by all samples
    area = in_set/s
    pass


def make_random_point():
    # radius of the circle
    circle_r = 2
    # center of the circle (x, y)
    circle_x = 0
    circle_y = 0

    # random angle
    alpha = 2 * math.pi * np.random.random()
    # random radius
    r = circle_r * math.sqrt(np.random.random())
    # calculating coordinates
    c = np.complex((r * math.cos(alpha) + circle_x), (r * math.sin(alpha) + circle_y ))
    return c

def draw_random_numbers(s):
    samples = np.zeros(s, dtype=np.csingle)
    for i in range(s):
        c = make_random_point()
        samples[i] = c
    return samples

if __name__ == '__main__':
    width = 400
    height = 400
    im = Image.new("RGB", (400, 400), color ="black")
    draw = ImageDraw.Draw(im)
    samples = draw_random_numbers(1000)
    for i in range(len(samples)):
        real = width/2 + np.real(samples[i]) * width/2
        imag = height/2 + np.imag(samples[i]) * height/2
        draw.point((real, imag), fill="white")
    im.show()