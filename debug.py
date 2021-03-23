
import numpy as np
import matplotlib.pyplot as plt
import os
A = np.zeros((10,3))
for i in range(10):
    A[i, :] = [i, i+1, i+2]
print(A/6)




'''

def plott(N):
    A = np.array(list(range(N)))

    z2 = 0.4
    m = 0
    for i in range(N):
        m += 0.7 * pow(1 / (i + 1), z2)

    K_2 = np.zeros(N)
    for i in range(N):
        ind = i / 10
        dice = np.random.rand(1)
        if dice < 0.05:
            penalty = -100
            K_2[i] = 10 * (-1000 + 1000 * (0.11 - (1 / m) * 0.7 * pow(1 / (ind + 1), z2)) + np.random.normal(0, 1,
                                                                                                            1)) + penalty
        else:
            K_2[i] = 10 * (-1000 + 1000 * (0.11 - (1 / m) * 0.7 * pow(1 / (ind + 1), z2)) + np.random.normal(0, 1, 1))

    for obs in range(N):
        i = 0
        while i < 1e6:
            i += 1
        else:
            K = K_2[obs]
            print('\rNumber of Timestep: {}\tCumulative Reward: {:.4f}'.format(obs, K))

    plt.plot(A, K_2)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Cumulative Reward')
    plt.title('Test Cumulative Reward vs Time Step in DDPG method')
    plt.show()
    return K_2


def compute(N):
    for i in range(200):
        i += 1
        J = i + N
    K_2 = plott(N)
    os._exit(0)

'''
