from calculate_area import *
import matplotlib.pyplot as plt

MANDEL_AREA = 1.5065918849  # Estimated value according to https://web.archive.org/web/20210715173755/https://www.foerstemann.name/dokuwiki/doku.php?id=numerical_estimation_of_the_area_of_the_mandelbrot_set_2012


def gen_square_primes():
    """ Generate an infinite sequence of prime numbers.
    """
    # Sieve of Eratosthenes
    # Code by David Eppstein, UC Irvine, 28 Feb 2002
    # http://code.activestate.com/recipes/117119/

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
    """Get the convergence data for increasing number of iterations (j)."""
    j_data = []
    n_simulations = 10

    for i in range(n_simulations):
        # Simulate
        num_it, res = simulate_j_convergence()
        j_data.append(res)
    return num_it, j_data


def get_s_convergence():
    """Get the convergence data for increasing number of samples (s)."""
    s_data = []
    n_simulattions = 10

    for i in range(n_simulattions):
        # Simulate
        print(f"simulation {i}")
        square_primes, res = simulate_s_convergence()
        s_data.append(res)
    return square_primes, s_data


def simulate_j_convergence():
    """Simulation (one trial) of the convergence of the Mandelbrot area for increasing number of iterations (j)."""
    # Init values
    num_it = [50, 100, 150, 200, 400, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    s = int(18769) # fixed sample size, must be square of prime number for orthogonal sampler
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
    """Plot the convergence of Mandelbrot area estimation for increasing number of iterations: Aj -> Ai. """
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


def simulate_s_convergence():
    """Simulation (one trial) of the convergence of the Mandelbrot area for increasing number of samples: Ak -> As."""
    # Init number of samples for orthogonal sampler - have to be square prime numbers!
    n = 9
    square_primes = generate_square_primes(n)
    assert n == len(square_primes)

    # Init sampling methods, number of iterations, results array
    num_it = [2000]
    sampling_methods = [pure_random_sampler, lh_sampler, orthogonal_sampler, antithetic_variate_sampler]
    res = np.zeros((n, len(sampling_methods)))

    # Generate approximations
    for sampler_i in range(len(sampling_methods)):
        sampler = sampling_methods[sampler_i]
        print(f"Sampler index: {sampler_i}")
        for k in range(n):
            n_samples = int(square_primes[k])

            # Approximate area
            mandelpoints_list = draw_mandel_samples(n_samples, sampler, num_it)
            for j in range(len(num_it)):
                area = approx_area(n_samples, mandelpoints_list[j])
                print(f"area: {area}: {n_samples}, {mandelpoints_list[j]}")
                # Store results
                res[k, sampler_i] = area

    return square_primes, res


def generate_square_primes(n):
    """Generate n square prime numbers to investigate convergence. Convert """
    generator = gen_square_primes()
    square_primes = np.zeros(n)
    values = np.arange(1, 5 * n + 1)
    j = 0
    for i in values:
        p = next(generator)
        if i % 5 == 0:
            square_primes[j] = p
            j += 1
    return square_primes


def plot_A_convergence(square_primes, data, data_name):
    """Plot the convergence of Mandelbrot area. """
    # Plot estimated true value as horizontal line
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
    plot_A_convergence(num_it, j_data, "j")
    square_primes, s_data = get_s_convergence()
    plot_A_convergence(square_primes, s_data, "k")
