import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    data = np.random.normal(0, 1, n)

    plt.hist(data, bins=10)
    plt.title("Normal(0,1) Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    return data


def uniform_histogram(n):
    data = np.random.uniform(0, 10, n)

    plt.hist(data, bins=10)
    plt.title("Uniform(0,10) Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    return data


def bernoulli_histogram(n):
    data = np.random.binomial(1, 0.5, n)

    plt.hist(data, bins=10)
    plt.title("Bernoulli(0.5) Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    return np.sum(data) / len(data)


def sample_variance(data):
    n = len(data)
    mean = sample_mean(data)
    return np.sum((data - mean) ** 2) / (n - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    sorted_data = np.sort(data)

    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    median = np.median(sorted_data)

    n = len(sorted_data)

    # Median-of-halves method (required by test)
    if n % 2 == 1:
        lower_half = sorted_data[:n // 2]
        upper_half = sorted_data[n // 2 + 1:]
    else:
        lower_half = sorted_data[:n // 2]
        upper_half = sorted_data[n // 2:]

    q1 = np.median(lower_half)
    q3 = np.median(upper_half)

    return minimum, maximum, median, q1, q3


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    n = len(x)
    mean_x = sample_mean(x)
    mean_y = sample_mean(y)

    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)

    return np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])
