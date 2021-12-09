import numpy as np
import matplotlib.pyplot as plt
from IPA import DTheta_estimator_fixed_theta


def decreasing_gain_example():
    gain = 0.01
    gains = []

    for n in range(5000):
        gain = gain * 0.999
        gains.append(gain)
    plt.title('Plot of decreasing gain for illustration')
    plt.ylabel('stepsize/gain')
    plt.plot(gains)


def stochastic_approximation(theta_0, stepsize, n_smaples, decreasing_stepsize=False):
    # setup parameters
    theta = theta_0
    S = 20
    h = 4
    b = 8
    p = 10
    N = 20
    scale = 15
    thetas = []
    n = int(computational_budget/n_smaples) # iterations

    for i in range(n):
        D_theta = 0
        for j in range(n_smaples):
            demand = np.random.exponential(scale, N)  # exponential demands
            D_theta += DTheta_estimator_fixed_theta(theta, theta, 1, demand, N, S, h, b, p)


        theta -= stepsize*D_theta/n_smaples
        thetas.append(theta)
        if decreasing_stepsize:
            stepsize *= 0.999
    return thetas


# thetas = stochastic_approximation(30, 0.01, 1)      # standard
# plt.plot(thetas)
# plt.figure()

#standard values
standard_theta = 30  # theta0
standard_stepsize = 0.01
standard_n_samples = 1
global computational_budget
computational_budget = 5000  # computational budget for 1 sample n equals iterations


# test different theta_0s
theta_0s = [10, 20, 30, 40]
n_samples = [1, 2, 5, 10]

fig, axis = plt.subplots(2, len(theta_0s))
for t, theta_0 in enumerate(theta_0s):
    axis[1, t].plot(stochastic_approximation(theta_0, 0.01, 1))
    axis[1, t].set_title('theta_0 = ' + str(theta_0))
    axis[1, t].set_xlabel('iterations (n)')
    axis[1, t].set_ylabel('Theta')

for s, samples in enumerate(n_samples):
    axis[0, s].plot(stochastic_approximation(30, 0.01, samples))
    axis[0, s].set_title('n_samples = ' + str(samples))
    axis[0, s].set_xlabel('iterations (n)')
    axis[0, s].set_ylabel('Theta')

fig.suptitle('computational budget = ' + str(computational_budget) + ' total iterations')

#
# gain sizes
fig, axis = plt.subplots(1, 2)
fixed_gains = [0.001, 0.005, 0.01, 0.02, 0.05]
fixed_gains.reverse()
decreasing_gains = [0.02, 0.05, 0.1, 0.2]
decreasing_gains.reverse()
for e, gain in enumerate(fixed_gains):
    axis[0].plot(stochastic_approximation(standard_theta, gain, standard_n_samples), label=gain, alpha=0.95)
    axis[0].set_title('fixed gain')
    axis[0].set_xlabel('iterations (n)')
    axis[0].set_ylabel('Theta')
    axis[0].legend()
for e, gain in enumerate(decreasing_gains):
    axis[1].plot(stochastic_approximation(standard_theta, gain, standard_n_samples, decreasing_stepsize=True), label=gain, alpha=0.95)
    axis[1].set_title('decreasing gain')
    axis[1].set_xlabel('iterations (n)')
    axis[1].set_ylabel('Theta')
    axis[1].legend()

plt.figure()
decreasing_gain_example()

plt.show()

