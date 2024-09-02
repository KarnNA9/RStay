import numpy as np
from scipy.stats import norm, expon, poisson, gamma

# Example 1: Maximum Likelihood Estimation for Normal Distribution
def mle_normal(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return mu, sigma

data_example1 = [1.2, 2.5, 1.7, 3.3, 4.1]
mu1, sigma1 = mle_normal(data_example1)
print("Example 1 - Normal Distribution:")
print("Data:", data_example1)
print("Estimated mean:", mu1)
print("Estimated standard deviation:", sigma1)
print()

# Example 2: Maximum Likelihood Estimation for Exponential Distribution
def mle_exponential(data):
    lam = 1 / np.mean(data)
    return lam

data_example2 = [0.5, 1.3, 2.7, 1.1, 0.8]
lam2 = mle_exponential(data_example2)
print("Example 2 - Exponential Distribution:")
print("Data:", data_example2)
print("Estimated lambda:", lam2)
print()

# Example 3: Maximum Likelihood Estimation for Poisson Distribution
def mle_poisson(data):
    lam = np.mean(data)
    return lam

data_example3 = [2, 4, 3, 5, 2]
lam3 = mle_poisson(data_example3)
print("Example 3 - Poisson Distribution:")
print("Data:", data_example3)
print("Estimated lambda:", lam3)
print()

# Example 4: Maximum Likelihood Estimation for Gamma Distribution
def mle_gamma(data):
    shape = (np.mean(data) / np.var(data)) ** 2
    scale = np.var(data) / np.mean(data)
    return shape, scale

data_example4 = [5, 10, 15, 20, 25]
shape4, scale4 = mle_gamma(data_example4)
print("Example 4 - Gamma Distribution:")
print("Data:", data_example4)
print("Estimated shape:", shape4)
print("Estimated scale:", scale4)
print()

# Example 5: Maximum Likelihood Estimation for Mixture of Normals (2 components)
def mle_mixture_of_normals(data):
    # Assuming equal weights for the two components
    mu1 = np.mean(data)
    sigma1 = np.std(data)
    mu2 = np.mean(data) + np.std(data)
    sigma2 = np.std(data)
    return mu1, sigma1, mu2, sigma2

data_example5 = [1.2, 2.5, 1.7, 3.3, 4.1, 10.2, 9.8, 11.5, 10.9]
mu1_5, sigma1_5, mu2_5, sigma2_5 = mle_mixture_of_normals(data_example5)
print("Example 5 - Mixture of Normals:")
print("Data:", data_example5)
print("Estimated mean (Component 1):", mu1_5)
print("Estimated standard deviation (Component 1):", sigma1_5)
print("Estimated mean (Component 2):", mu2_5)
print("Estimated standard deviation (Component 2):", sigma2_5)
