import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import qmc


def approx_area(n_samples, n_mandelbrot_samples):
    """Approximate the area as ratio of samples within Mandelbrot set and total samples drawn from circle with radius 2."""
    circle_r = 2  # circle radius is always 2
    total_area = np.pi * circle_r * circle_r
    frac = n_mandelbrot_samples / n_samples
    mandelbrot_area = total_area * frac
    return mandelbrot_area


def lh_sampler(s):
    """Draw 2 random number samples of size s from Latin Hypercube sampler. """
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=s)
    return sample[:, 0], sample[:, 1]


def antithetic_variate_sampler(s):
    """Draw 2 vectors of antithetic variates."""
    rand1 = np.zeros(s)
    rand2 = np.zeros(s)
    for i in range(s):
        if i % 2 == 0:
            rand1[i] = np.random.random()
            rand2[i] = np.random.random()
        else:
            rand1[i] = 1 - rand1[i-1]
            rand2[i] = 1 - rand2[i-1]
    return rand1, rand2


def orthogonal_sampler(s):
    """Draw 2 random number samples of size s from orthogonal sampler. NB: s has to be the square root of a prime number."""
    sampler = qmc.LatinHypercube(d=2, strength=2)
    sample = sampler.random(n=s)
    return sample[:, 0], sample[:, 1]


def pure_random_sampler(s):
    """Draw 2 random number samples of size s from pure random generator. """
    rand1 = np.random.random(s)
    rand2 = np.random.random(s)
    return rand1, rand2


def antithetic_variate_sampler(s):
    """Draw 2 vectors of antithetic variates."""
    rand1 = np.zeros(s)
    rand2 = np.zeros(s)
    for i in range(s):
        if i % 2 == 0:
            rand1[i] = np.random.random()
            rand2[i] = np.random.random()
        else:
            rand1[i] = 1 - rand1[i-1]
            rand2[i] = 1 - rand2[i-1]
    return rand1, rand2


def convert_to_circle(x, y):
    """Convert point (x,y) to a complex circle coordinate."""
    # center of the circle (x, y)
    circle_x = 0
    circle_y = 0
    circle_r = 2  # radius is always 2

    # random angle
    alpha = 2 * math.pi * x
    # random radius
    r = circle_r * math.sqrt(y)
    # calculating coordinates
    c = complex((r * math.cos(alpha) + circle_x), (r * math.sin(alpha) + circle_y))
    return c


def count_stable_evals(c, num_iterations, mandelpoints_list):
    """For each iteration in num_iterations, incrementally check if c is still stable. Store the count of
    stable evaluations in mandelpoints_list."""
    stable_list = []
    z = 0

    # Incrementally check if current c is still stable after num iterations
    for i, num in enumerate(num_iterations):
        if num != num_iterations[0]:
            num = num_iterations[i] - num_iterations[i-1]

        for _ in range(num):
            if abs(z) > 2:
                stable_list.append(False)
                break
            
            z = z ** 2 + c

        if abs(z) <= 2:
            stable_list.append(True)

    # Count number of stable points and save count in mandelpoints list
    for i, stable in enumerate(stable_list):
        if stable:
            mandelpoints_list[i] += 1

    return(mandelpoints_list)


def draw_mandel_samples(s, sampler, num_it):
    """Given a sample size s, a sampling method sampler and number of iterations num_it;
    return a list that holds the number of stable points for each number of iterations per sample point."""
    mandelpoints_list = [0] * len(num_it)
    # Get two random number arrays
    rand1, rand2 = sampler(s)
    for j in range(s):
        # Convert random numbers to circle points
        c = convert_to_circle(rand1[j], rand2[j])
        mandelpoints_list = count_stable_evals(c, num_it, mandelpoints_list)

    return mandelpoints_list


def draw_mandelbrot(mandel_samples):
    """Draw a black-and-white image of the Mandelbrot set. """
    width = 1000
    height = 1000
    im = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(im)

    for i in range(len(mandel_samples)):
        real = width / 2 + np.real(mandel_samples[i]) * width / 4
        imag = height / 2 + np.imag(mandel_samples[i]) * height / 4  # should be height/(2*radius)
        draw.point((real, imag), fill="black")
    im.show()
