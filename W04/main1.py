import numpy as np
from scipy.stats import norm, expon, poisson, gamma
import matplotlib.pyplot as plt

# Example 1: Maximum Likelihood Estimation for Normal Distribution
global rep 
rep = 1
while rep == 1:
    def mle_normal(data):
        mu = np.mean(data)
        sigma = np.std(data)
        return mu, sigma

    # np.random.seed(0)
    data_example1 = [1.2, 2.5, 1.7, 3.3, 4.1] # np.random.uniform(1, 4.5, 50) # 
    mu1, sigma1 = mle_normal(data_example1)
    print(" - Example 1 - Normal Distribution:")
    print(" - Data:", data_example1)
    print(" - Estimated mean (mu1):", mu1)
    print(" - Estimated standard deviation (sigma1):", sigma1)
    print()

    # histograma
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data_example1, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Histograma para data_example1')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    # distribución normal estimada
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu1, sigma1)
    plt.subplot(1, 2, 2)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Distribución Normal Estimada')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de probabilidad')
    plt.tight_layout()
    plt.show()
    rep = int(input("\n Presione 1 para repetir, 2 para seguir: "))

rep = 1
while rep == 1:
# Example 2: Maximum Likelihood Estimation for Exponential Distribution
    def mle_exponential(data):
        lam = 1 / np.mean(data)
        return lam

    data_example2 = [0.5, 1.3, 2.7, 1.1, 0.8] # np.random.uniform(0.5, 3, 50) # 
    lam2 = mle_exponential(data_example2)
    print(" - Example 2 - Exponential Distribution:")
    print(" - Data:", data_example2)
    print(" - Estimated lambda:", lam2)
    print()

    # histograma
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data_example2, bins=20, edgecolor='black', alpha=0.7, density=True)
    plt.title('Histograma para data_example2')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    # distribución exponencial estimada
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = expon.pdf(x, scale=1/lam2)
    plt.subplot(1, 2, 2)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Distribución Exponencial Estimada')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de probabilidad')
    plt.tight_layout()
    plt.show()
    rep = int(input("\n Presione 1 para repetir, 2 para seguir: "))
    
rep = 1
while rep == 1:
    # Example 3: Maximum Likelihood Estimation for Poisson Distribution
    def mle_poisson(data):
        lam = np.mean(data)
        return lam

    data_example3 = [2, 4, 3, 5, 2] # np.random.uniform(2, 5, 50) # 
    lam3 = mle_poisson(data_example3)
    print(" - Example 3 - Poisson Distribution:")
    print(" - Data:", data_example3)
    print(" - Estimated lambda:", lam3)
    print()

    # histograma
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data_example2, bins=20, edgecolor='black', alpha=0.7, density=True)
    plt.title('Histograma para data_example3')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    # distribución poisson estimada
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = expon.pdf(x, scale=1/lam3)
    plt.subplot(1, 2, 2)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Distribución Exponencial Estimada')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de probabilidad')
    plt.tight_layout()
    plt.show()
    rep = int(input("\n Presione 1 para repetir, 2 para seguir: "))
    
rep = 1
while rep == 1:
    # Example 4: Maximum Likelihood Estimation for Gamma Distribution
    def mle_gamma(data):
        shape = (np.mean(data) / np.var(data)) ** 2
        scale = np.var(data) / np.mean(data)
        return shape, scale

    data_example4 = [5, 10, 15, 20, 25] # np.random.uniform(5, 20, 50) # 
    shape4, scale4 = mle_gamma(data_example4)
    print(" - Example 4 - Gamma Distribution:")
    print(" - Data:", data_example4)
    print(" - Estimated shape:", shape4)
    print(" - Estimated scale:", scale4)
    print()

    # histograma
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data_example4, bins=15, edgecolor='black', alpha=0.7, density=True)
    plt.title('Histograma para data_example4')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    # distribución gamma estimada
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = gamma.pdf(x, shape4, scale=scale4)
    plt.subplot(1, 2, 2)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Distribución Gamma Estimada')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de Probabilidad')
    plt.tight_layout()
    plt.show()
    rep = int(input("\n Presione 1 para repetir, 2 para seguir: "))
    
rep = 1
while rep == 1:
# Example 5: Maximum Likelihood Estimation for Mixture of Normals (2 components)
    def mle_mixture_of_normals(data):
        # Assuming equal weights for the two components
        mu1 = np.mean(data)
        sigma1 = np.std(data)
        mu2 = np.mean(data) + np.std(data)
        sigma2 = np.std(data)
        return mu1, sigma1, mu2, sigma2

    data_example5 = [1.2, 2.5, 1.7, 3.3, 4.1, 10.2, 9.8, 11.5, 10.9] #  np.random.uniform(1, 12, 50) # 
    mu1_5, sigma1_5, mu2_5, sigma2_5 = mle_mixture_of_normals(data_example5)
    print(" - Example 5 - Mixture of Normals:")
    print(" - Data:", data_example5)
    print(" - Estimated mean (Component 1):", mu1_5)
    print(" - Estimated standard deviation (Component 1):", sigma1_5)
    print(" - Estimated mean (Component 2):", mu2_5)
    print(" - Estimated standard deviation (Component 2):", sigma2_5)

    # Histograma
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data_example5, bins=10, edgecolor='black', alpha=0.7, density=True)
    plt.title('Histograma de Datos')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    # Rango de valores para las curvas
    xmin, xmax = min(data_example5) - 1, max(data_example5) + 1
    x = np.linspace(xmin, xmax, 1000)
    # componentes normales
    p1 = norm.pdf(x, mu1_5, sigma1_5)
    p2 = norm.pdf(x, mu2_5, sigma2_5)
    # mezcla de las dos normales
    p_mixture = 0.5*p1 + 0.5*p2
    plt.subplot(1, 2, 2)
    plt.plot(x, p1, 'r--', label='Componente 1', linewidth=2)
    plt.plot(x, p2, 'b--', label='Componente 2', linewidth=2)
    plt.plot(x, p_mixture, 'k-', label='Mezcla', linewidth=2)
    plt.title('Mezcla de Dos Normales')
    plt.xlabel('Valor')
    plt.ylabel('Densidad de Probabilidad')
    plt.legend()
    plt.tight_layout()
    plt.show()
    rep = int(input("\n Presione 1 para repetir, 2 para seguir: "))