import numpy as np
import matplotlib
matplotlib.use("Agg")  # CI-safe backend (no GUI)

from stats_lab import (
    normal_histogram,
    uniform_histogram,
    bernoulli_histogram,
    sample_mean,
    sample_variance,
    order_statistics,
    sample_covariance,
    covariance_matrix,
)

# -----------------------------------
# Question 1 – Generate & Plot Histograms
# (We grade by checking the generated sample properties.)
# -----------------------------------

def test_normal_histogram_generation():
    np.random.seed(0)
    data = normal_histogram(10000)

    assert isinstance(data, (list, np.ndarray))
    data = np.asarray(data)

    assert data.shape == (10000,)
    assert abs(sample_mean(data)) < 0.1
    assert abs(sample_variance(data) - 1.0) < 0.1


def test_uniform_histogram_generation():
    np.random.seed(0)
    data = uniform_histogram(10000)

    assert isinstance(data, (list, np.ndarray))
    data = np.asarray(data)

    assert data.shape == (10000,)
    assert data.min() >= 0
    assert data.max() <= 10
    assert abs(sample_mean(data) - 5.0) < 0.1
    # Optional: could also check variance ≈ (10-0)^2/12 = 8.333...


def test_bernoulli_histogram_generation():
    np.random.seed(0)
    data = bernoulli_histogram(10000)

    assert isinstance(data, (list, np.ndarray))
    data = np.asarray(data)

    assert data.shape == (10000,)
    assert set(np.unique(data)).issubset({0, 1})
    assert abs(sample_mean(data) - 0.5) < 0.05


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def test_sample_mean():
    data = np.array([1, 2, 3, 4, 5])
    assert sample_mean(data) == 3


def test_sample_variance():
    data = np.array([1, 2, 3, 4, 5])
    assert np.isclose(sample_variance(data), 2.5)


# -----------------------------------
# Question 3 – Order Statistics (assert Q1 and Q3)
# -----------------------------------

def test_order_statistics():
    data = np.array([5, 1, 3, 2, 4])
    minimum, maximum, median, q1, q3 = order_statistics(data)

    assert minimum == 1
    assert maximum == 5
    assert median == 3
    assert q1 == 2
    assert q3 == 4


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def test_sample_covariance():
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
