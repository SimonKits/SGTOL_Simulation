from Stoch_approx_g import stochastic_approximation
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np


#standard values
standard_theta = 30  # theta0
standard_stepsize = 0.01
standard_n_samples = 1
# global computational_budget
# computational_budget = 5000  # computational budget for 1 sample n equals iterations
n_runs = 500

thetas_ls = []
for run in range(n_runs):
    thetas_ls.append(stochastic_approximation(standard_theta, standard_stepsize, standard_n_samples,
                                              decreasing_stepsize=False, computational_budget=3000))

means = []
for thetas in thetas_ls:
    # means.append(np.mean(thetas[:100]))
    means.append(thetas[-1])
plt.hist(means, bins=20)
jarque_bera = st.jarque_bera(means)

print(jarque_bera)
print(f'The p-value of the Jarque-Bera test ({jarque_bera.pvalue}) > 0.05, therefor we accept the Null hypothesis and '
      f'conclude that the distribution is normal.')
print()
print(f'Mean = {np.mean(means)}, var= {np.var(means)}')
plt.show()