import numpy as np
import matplotlib.pyplot as plt

def matrix_factorization(R, P, Q, K, steps=5000, gamma=0.0002, lambda_=0.02):
    Q = Q.T
    # x = []
    # y = []
    for step in xrange(steps):
        for i in xrange(R.shape[0]):
            for j in xrange(R.shape[1]):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                    for k in xrange(K):
                        gradient_p = -2 * eij * Q[k][j] + lambda_ * P[i][k]
                        gradient_q = -2 * eij * P[i][k] + lambda_ * Q[k][j]
                        if P[i][k] - gamma * gradient_p >= 0:
                            P[i][k] = P[i][k] - gamma * gradient_p
                        if Q[k][j] - gamma * gradient_q >= 0:
                            Q[k][j] = Q[k][j] - gamma * gradient_q
        eR = np.dot(P, Q)
        e = 0
        for i in xrange(R.shape[0]):
            for j in xrange(R.shape[1]):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - eR[i][j], 2)
                    for k in xrange(K):
                        e = e + (lambda_ / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        # x.append(step)
        # y.append(e)
        if e < 0.001:
            break
    # plt.plot(x, y)
    # plt.show()
    return P, Q.T
