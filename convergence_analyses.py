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
    j_data = []
    n_simulations = 10

    for i in range(n_simulations):
        # Generate data
        num_it, res = simulate_j_convergence()
        j_data.append(res)
        # Save to pandas
        df = pd.DataFrame(res, columns=["Pure Random", "Latin Hypercube", "Orthogonal", "antithetic_variate_sampler"])
        df["Number of Iterations"] = num_it
    df.to_csv(filepath, index=False)
    # else:
    #     # Read from csv
    #     df = pd.read_csv(filepath)
    #     num_it = df["Number of Iterations"]
    #     num_it = num_it.to_numpy()
    #     res = df[["Pure Random", "Latin Hypercube", "Orthogonal", "antithetic_variate_sampler"]]
    #     res = res.to_numpy()
    return num_it, j_data


def get_s_convergence():
    filepath = "convergence-s-data-large.csv"
    # if not os.path.isfile(filepath):
    s_data = []
    n_simulattions = 10

    for i in range(n_simulattions):
        # Simulate
        print(f"simulation {i}")
        samples, square_primes, res = simulate_s_convergence()
        s_data.append(res)
        # Save to pandas
        df = pd.DataFrame(res, columns=["Pure Random", "Latin Hypercube", "Orthogonal", "antithetic_variate_sampler"])
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
    #     res = df[["Pure Random", "Latin Hypercube", "Orthogonal", "antithetic_variate_sampler"]]
    #     res = res.to_numpy()
    return samples, square_primes, s_data


def simulate_j_convergence():
    # Init values
    num_it = [50, 100, 150, 200, 400, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    s = int(18769)
    sampling_methods = [pure_random_sampler, lh_sampler, orthogonal_sampler, antithetic_variate_sampler]
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


def plot_j_convergence(num_it, j_data):
    """Plot the convergence of Mandelbrot area estimation for increasing number of iterations"""
    plt.title("Convergence of $A_{j,s}$ to $A_{i,s}$, $j < i$  $(s=18769)$")

    random_data = []
    lh_data = []
    ot_data = []
    av_data = []
    for res in j_data:
        random_data.append(abs(res[-1, 0] - res[:-1, 0]))
        lh_data.append(abs(res[-1, 1] - res[:-1, 1]))
        ot_data.append(abs(res[-1, 2] - res[:-1, 2]))
        av_data.append(abs(res[-1, 3] - res[:-1, 3]))
    
    random_mean = np.mean(random_data, axis=0)
    random_std = np.std(random_data, axis=0)
    
    lh_mean = np.mean(lh_data, axis=0)
    lh_std = np.std(lh_data, axis=0)

    ot_mean = np.mean(ot_data, axis=0)
    ot_std = np.std(ot_data, axis=0)

    av_mean = np.mean(av_data, axis=0)
    av_std = np.std(av_data, axis=0)

    plt.plot(num_it[:-1], random_mean, "o-", color="tab:blue", label=f"Pure random sampling")
    plt.plot(num_it[:-1], lh_mean, "o-", color="tab:orange", label=f"Latin Hypercube sampling")
    plt.plot(num_it[:-1], ot_mean, "o-", color="tab:green", label="Orthogonal sampling")
    plt.plot(num_it[:-1], av_mean, "o-", color="tab:red", label="Antithetic variate sampling")

    plt.fill_between(num_it[:-1], random_mean+random_std, random_mean-random_std, alpha=0.35, color="tab:blue", linewidth=0)
    plt.fill_between(num_it[:-1], lh_mean+lh_std, lh_mean-lh_std, alpha=0.35, color="tab:orange", linewidth=0)
    plt.fill_between(num_it[:-1], ot_mean+ot_std, ot_mean-ot_std, alpha=0.35, color="tab:green", linewidth=0)
    plt.fill_between(num_it[:-1], av_mean+av_std, av_mean-av_std, alpha=0.35, color="tab:red", linewidth=0)

    plt.axhline(0, color="black", alpha=0.5, linestyle="dotted")

    plt.ylabel("$\Vert A_{j,s} - A_{i,s} \Vert$")
    plt.xlabel("Number of Iterations $(j)$")
    plt.legend()
    plt.savefig("figures/convergence_i-j.png")
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
    s = [10e2, 10e3, 10e4, 20e4, 40e4, 60e4, 80e4, 10e5, 20e5]  # must be square of prime number for othogonal sampler
    
    # Init number of samples for orthogonal sampler - have to be square prime numbers!
    square_primes = generate_square_primes(s)
    assert len(s) == len(square_primes)

    # Init sampling methods, number of iterations, results array
    num_it = [2000]
    sampling_methods = [pure_random_sampler, lh_sampler, orthogonal_sampler, antithetic_variate_sampler]
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


def plot_A_convergence(square_primes, data, data_name):
    plt.axhline(MANDEL_AREA, color="black", linestyle="dashed", label="True Value")

    random_data = []
    lh_data = []
    ot_data = []
    av_data = []

    for res in data:
        random_data.append(res[:, 0])
        lh_data.append(res[:, 1])
        ot_data.append(res[:, 2])
        av_data.append(res[:, 3])

    random_mean = np.mean(random_data, axis=0)
    random_std = np.std(random_data, axis=0)
    
    lh_mean = np.mean(lh_data, axis=0)
    lh_std = np.std(lh_data, axis=0)

    ot_mean = np.mean(ot_data, axis=0)
    ot_std = np.std(ot_data, axis=0)

    av_mean = np.mean(av_data, axis=0)
    av_std = np.std(av_data, axis=0)

    plt.plot(square_primes, random_mean, "o-", color="tab:blue", label="Pure random sampling")
    plt.plot(square_primes, lh_mean, "o-", color="tab:orange", label="Latin Hypercube sampling")
    plt.plot(square_primes, ot_mean, "o-", color="tab:green", label="Orthogonal sampling")
    plt.plot(square_primes, av_mean, "o-", color="tab:red", label="Antithetic variate sampling")

    plt.fill_between(square_primes, random_mean+random_std, random_mean-random_std, alpha=0.35, color="tab:blue", linewidth=0)
    plt.fill_between(square_primes, lh_mean+lh_std, lh_mean-lh_std, alpha=0.35, color="tab:orange", linewidth=0)
    plt.fill_between(square_primes, ot_mean+ot_std, ot_mean-ot_std, alpha=0.35, color="tab:green", linewidth=0)
    plt.fill_between(square_primes, av_mean+av_std, av_mean-av_std, alpha=0.35, color="tab:red", linewidth=0)

    plt.legend()
    
    if data_name == "k":
        plt.xlabel("Number of Samples $(k)$")
        plt.title("Convergence of Mandelbrot Area Approximation\nfor Different Sampling Methods $(j=2000)$")
    elif data_name == "j":
        plt.xlabel("Number of Iterations $(j)$")
        plt.title("Convergence of Mandelbrot Area Approximation\nfor Different Sampling Methods $(k=18769)$")

    plt.ylabel("Estimated Area of Mandelbrot Set")
    plt.savefig(f"figures/convergence_a_{data_name}.png")
    plt.show()

    # plot std
    plt.plot(square_primes, random_std, "o-", color="tab:blue", label="Pure random sampling")
    plt.plot(square_primes, lh_std, "o-", color="tab:orange", label="Latin Hypercube sampling")
    plt.plot(square_primes, ot_std, "o-", color="tab:green", label="Orthogonal sampling")
    plt.plot(square_primes, av_std, "o-", color="tab:red", label="Antithetic variate sampling")

    plt.ylim(bottom=0)
    
    plt.legend()
    if data_name == "k":
        plt.xlabel("Number of Samples $(k)$")
        plt.title("Standard Deviation of estimated Area over 10 Simulations\nfor different Sampling Methods $(j=2000)$")
    elif data_name == "j":
        plt.xlabel("Number of Iterations $(j)$")
        plt.title("Standard Deviation of estimated Area over 10 Simulations\nfor different Sampling Methods $(k=18769)$")
    plt.ylabel("Standard Deviation")
    plt.savefig(f"figures/standard_deviation_{data_name}.png")
    plt.show()


if __name__ == '__main__':
    num_it, j_data = get_j_convergence()
    plot_j_convergence(num_it, j_data)
    #plot_A_convergence(num_it, j_data, "j")
    #samples, square_primes, s_data = get_s_convergence()
    # plot_s_convergence(samples, square_primes, s_data)
    #plot_A_convergence(square_primes, s_data, "k")
