# Descriptive Statistics Lab (Autograded)

## What students implement
Edit `stats_lab.py` and implement:

### Q1 — Generate & plot histograms (10 bins) and return the sample
- `normal_histogram(n)` must sample **Normal(0,1)**
- `uniform_histogram(n)` must sample **Uniform(0,10)**
- `bernoulli_histogram(n)` must sample **Bernoulli(0.5)**

Each function must:
- generate `n` samples using NumPy
- plot a histogram with **10 bins**
- label axes and add a title
- call `plt.show()`
- **return** the generated data (as list or NumPy array)

### Q2 — Sample mean & variance
- `sample_mean(data)`
- `sample_variance(data)` must use **n-1** in denominator

### Q3 — Order statistics
Return `(min, max, median, q1, q3)`.
For the dataset `[5,1,3,2,4]`, the autograder expects **Q1=2** and **Q3=4**.

### Q4 — Sample covariance
Use **n-1** in denominator.

### Q5 — Covariance matrix
Return a 2×2 matrix:
