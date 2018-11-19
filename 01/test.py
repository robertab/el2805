#/usr/bin/python3


import numpy as np
from pprint import pprint

def main():
    P = []
    for _ in range(4):
        P.append([None, None, None, None])

    P[0][0] = (1/3, 0, 0)
    P[0][1] = (1/3, 1, 0)
    P[0][2] = (1/3, 2, 0)
    P[0][3] = (0, 0, 0)

    P[1][0] = (1/3, 0, 0)
    P[1][1] = (1/3, 1, 0)
    P[1][2] = (0, 0, 0)
    P[1][3] = (1/3, 3, 1)

    P[2][0] = (1/3, 0, 0)
    P[2][1] = (0, 0, 0)
    P[2][2] = (1/3, 2, 0)
    P[2][3] = (1/3, 3, 1)

    P[3][1] = (0, 0, 0)
    P[3][0] = (0, 0, 0)
    P[3][2] = (0, 0, 0)
    P[3][3] = (1, 3, 1)
    # Initial policy
    policy = np.random.choice(3, size=4)
    pprint(pi)
    pprint(P)

    v = np.zeros((1, 4))
    for s in range(4):
        pa = policy[s]
        v[s] = sum([p * (r + v[s_new]) for p, s_new, r in P[s][pa]])



if __name__=='__main__':
    main()
