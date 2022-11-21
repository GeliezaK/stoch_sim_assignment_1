import os.path

import matplotlib.pyplot as plt
from calculate_area import *
import pandas as pd
import os.path

MANDEL_AREA = 1.5065918849  # Estimated value according to https://web.archive.org/web/20210715173755/https://www.foerstemann.name/dokuwiki/doku.php?id=numerical_estimation_of_the_area_of_the_mandelbrot_set_2012


def gen_square_primes():
    """ Generate an infinite sequence of prime numbers.
    """
    # Sieve of Eratosthenes
    # Code by David Eppstein, UC Irvine, 28 Feb 2002
    # http://code.activestate.com/recipes/117119/

    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}

    # The running integer that's checked for primeness
    q = 2

    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            #
            yield q**2
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next
            # multiples of its witnesses to prepare for larger
            # numbers
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]

        q += 1

def get_j_convergence():
    filepath = "convergence-j-stable2_test.csv"

    # if not os.path.isfile(filepath):
    # Generate data
    num_it, res = simulate_j_convergence()
    # Save to pandas
    df = pd.DataFrame(res, columns=["Pure Random", "Latin Hypercube", "Orthogonal"])
    # df = pd.DataFrame(res, columns=["Pure Random", "Latin Hypercube"])
    df["Number of Iterations"] = num_it
    df.to_csv(filepath, index=False)
    # else:
    #     # Read from csv
    #     df = pd.read_csv(filepath)
    #     num_it = df["Number of Iterations"]
    #     num_it = num_it.to_numpy()
    #     # res = df[["Pure Random", "Latin Hypercube", "Orthogonal"]]
    #     res = df[["Pure Random", "Latin Hypercube"]]
    #     res = res.to_numpy()
    return num_it, res

def get_s_convergence():
    filepath = "convergence-s-data-large.csv"
    # if not os.path.isfile(filepath):
    # Simulate
    samples, square_primes, res = simulate_s_convergence()
    # Save to pandas
    df = pd.DataFrame(res, columns=["Pure Random", "Latin Hypercube", "Orthogonal"])
    df["samples"] = samples
    df["square-prime-samples"] = square_primes
    df.to_csv(filepath, index=False)
    # else :
    #     # Read from csv
    #     df = pd.read_csv(filepath)
    #     samples = df["samples"]
    #     samples = samples.to_numpy()
    #     square_primes = df["square-prime-samples"]
    #     square_primes = square_primes.to_numpy()
    #     res = df[["Pure Random", "Latin Hypercube", "Orthogonal"]]
    #     res = res.to_numpy()
    return samples, square_primes, res

def simulate_j_convergence():
    # Init values
    num_it = [50, 100, 150, 200, 400, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]
    s = int(18769)
    sampling_methods = [pure_random_sampler, lh_sampler, orthogonal_sampler]
    res = np.zeros((len(num_it), len(sampling_methods)))

    # Generate approximations
    for sampler_i, sampler in enumerate(sampling_methods):
        # Approximate area
        mandelpoints_list = draw_mandel_samples(s, sampler, num_it)

        for j in range(len(num_it)):
            area = approx_area(s, mandelpoints_list[j])
            print(f"area: {area}: {s}, {mandelpoints_list[j]}")

            # Store results
            res[j, sampler_i] = area

    return num_it, res

def plot_j_convergence(num_it, res):
    """Plot the convergence of Mandelbrot area estimation for increasing number of iterations"""
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.title(f"s = 200000")
    plt.plot(num_it[:-1], abs(res[-1, 0] - res[:-1, 0]), "bo-", label=f"Pure random sampling")
    plt.plot(num_it[:-1], abs(res[-1, 1] - res[:-1, 1]), "go-", label=f"Latin Hypercube sampling")
    plt.axhline(0, color="black", alpha=0.5, linestyle="dotted")
    plt.ylabel("Absolute Error $\Vert A_{i,s} - A_{j,s} \Vert$")
    plt.xlabel("j, Number of Iterations")
    plt.legend()
    ax1 = plt.gca()
    ymin1, ymax1 = plt.ylim()
    plt.subplot(122)
    plt.plot(num_it[:-1], abs(res[-1, 2] - res[:-1, 2]), "co-", label="Orthogonal sampling")
    plt.axhline(0, color="black", alpha=0.5, linestyle="dotted")
    ax2 = plt.gca()
    ymin2, ymax2 = plt.ylim()
    plt.xlabel("j, Number of Iterations")
    plt.legend()
    plt.title("s = 18769")
    # Set axes equal
    ax1.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
    ax2.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
    plt.suptitle(f"Convergence of Mandelbrot Area Approximation, i= {num_it[-1]}")
    plt.savefig("Convergence-j.png")
    plt.show()



def plot_s_convergence(s, square_primes, res):
    """Plot the convergence of Mandelbrot area estimation for increasing sample sizes s"""
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.plot(square_primes[:-1], abs(res[-1, 0] - res[:-1, 0]), "bo-", label=f"Pure Random Sampling")
    plt.plot(square_primes[:-1], abs(res[-1, 1] - res[:-1, 1]), "go-", label=f"Latin Hypercube Sampling")
    plt.axhline(0, color="black", alpha=0.5, linestyle="dotted")
    plt.ylabel("Absolute Error $\Vert A_{i,k} - A_{i,s} \Vert$")
    plt.xlabel("k, Number of Samples Drawn")
    plt.legend()
    ax1 = plt.gca()
    ymin1, ymax1 = plt.ylim()
    plt.subplot(122)
    plt.plot(square_primes[:-1], abs(res[-1, 2] - res[:-1, 2]), "co-", label="Orthogonal Sampling")
    plt.axhline(0, color="black", alpha=0.5, linestyle="dotted")
    ax2 = plt.gca()
    ymin2, ymax2 = plt.ylim()
    plt.xlabel("k, Number of Samples Drawn")
    plt.legend()
    # Set axes equal
    ax1.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
    ax2.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
    plt.suptitle(f"Convergence of Mandelbrot Area Approximation, i= 2000")
    plt.savefig("Convergence-s.png")
    plt.show()


def simulate_s_convergence():
    # Init number of samples
    s = [10e2, 10e3, 10e4, 20e4, 40e4, 60e4, 80e4, 10e5, 20e5, 40e5]  # must be square of prime number for othogonal sampler

    # Init number of samples for orthogonal sampler - have to be square prime numbers!
    square_primes = generate_square_primes(s)
    assert len(s) == len(square_primes)

    # Init sampling methods, number of iterations, results array
    num_it = [2000]
    sampling_methods = [pure_random_sampler, lh_sampler, orthogonal_sampler]
    res = np.zeros((len(s), len(sampling_methods)))

    # Generate approximations
    for sampler_i in range(len(sampling_methods)):
        sampler = sampling_methods[sampler_i]
        for k in range(len(s)):
            n_samples = int(square_primes[k])
            print(f"Sampler index: {sampler_i}, k : {s[k]}")

            # Approximate area
            mandelpoints_list = draw_mandel_samples(n_samples, sampler, num_it)
            for j in range(len(num_it)):
                area = approx_area(n_samples, mandelpoints_list[j])
                print(f"area: {area}: {n_samples}, {mandelpoints_list[j]}")
                # Store results
                res[k, sampler_i] = area

    return s, square_primes, res


def generate_square_primes(s):
    """Generate s square prime numbers to investigate convergence."""
    generator = gen_square_primes()
    square_primes = np.zeros(len(s))
    values = np.arange(1, 5 * len(s) + 1)
    j = 0
    for i in values:
        p = next(generator)
        if i % 5 == 0:
            square_primes[j] = p
            j += 1
    return square_primes


def plot_A_convergence(samples, square_primes, res):
    plt.axhline(MANDEL_AREA, color="black", linestyle="dashed", label="True Value")
    plt.plot(square_primes, res[:, 0], "bo-", label="Pure Random Sampling")
    plt.plot(square_primes, res[:, 1], "go-", label="Latin Hypercube Sampling")
    plt.plot(square_primes, res[:,2], "co-", label="Orthogonal Sampling")
    plt.legend()
    plt.xlabel("k, Number of Samples")
    plt.ylabel("Estimated Area of Mandelbrot Set")
    plt.title("Convergence of Mandelbrot Area Approximation \n for Different Sampling Methods")
    plt.savefig("convergence-a.png")
    plt.show()


if __name__ == '__main__':
    num_it, res = get_j_convergence()
    plot_j_convergence(num_it, res)
    samples, square_primes, res = get_s_convergence()
    plot_s_convergence(samples, square_primes, res)
    plot_A_convergence(samples, square_primes, res)
