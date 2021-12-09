import numpy as np
import matplotlib.pyplot as plt
import random
import time


def Simulation(theta, inventory, S, h, b, p, N, demand):
    cost = 0
    for i in range(N):
        production = min(S,max(theta-inventory+demand[i],0))
        cost += h*max(inventory,0) + b*-min(inventory-demand[i],0) + p*production
        inventory = inventory - demand[i] + production
    return cost/N


if __name__=="__main__":
    theta = 10      # theta_0
    inventory = 10
    S = 20
    h = 4
    b = 8
    p = 10
    N = 20
    mu = 15
    n = 100
    demand = np.random.exponential(mu,[n,N])


    stepsize = 0.1
    begin, end = 10, 40
    J = np.zeros(int((end-begin)/stepsize))
    thetas = []
    for i in range(int((end-begin)/stepsize)):
        results = np.zeros(n)
        for j in range(n):
            results[j] = Simulation(theta, inventory, S, h, b, p, N, demand[j])
        thetas.append(theta)
        theta += stepsize
        inventory += stepsize
        J[i] = np.mean(results)
    plt.plot(thetas, J)
    plt.show()

