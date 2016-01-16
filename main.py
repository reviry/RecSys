import nmf2
from completion import compliment
import numpy as np

if __name__ == '__main__':
    R = [
          [5,3,0,1],
          [4,0,0,1],
          [1,1,0,5],
          [1,0,0,4],
          [0,1,5,4],
        ]

    M = [
          [1,1,0.5,1],
          [1,0.5,0.5,1],
          [1,1,0.5,1],
          [1,0.5,0.5,1],
          [0.5,1,1,1],
        ]

    R = np.array(R)
    M = np.array(M)

    R = compliment(R)

    K = 2

    P = np.random.rand(R.shape[0], K)
    Q = np.random.rand(R.shape[1], K)

    # nP, nQ = nmf1.matrix_factorization(R, P, Q, K)
    nP, nQ = nmf2.matrix_factorization(R, M, P, Q, K)
    nR = np.dot(nP, nQ.T)

    print R
    print nP
    print nQ
    print nR
