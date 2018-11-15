#/usr/bin/python3

"""
Problem 1 - The Maze and the Random Minotaur

Author: Robert Siwerz, Christopher Dahlén
Course: EL2805 - Reinforcement learning



Description:

(a) Formulate the problem as an MDP.
(b) Solve the problem, and illustrate an optimal policy for T = 15. 2 Plot the maximal probability
    of exiting the maze as a function of T . Is there a difference if the minotaur is allowed to
    stand still? If so, why?
(c) Assume now that your life is geometrically distributed with mean 30. Modify the problem
    so as to derive a policy minimising the expected time to exit the maze. Motivate your new
    problem formulation (model). Estimate the probability of getting out alive using this policy
    by simulating 10 000 games.
"""

""" Task (a) """
import numpy as np
import matplotlib.pyplot as plt

def get_player_transitions():
    p = np.zeros((30,30))
    thirds = [(0,0),(0,1),(0,6),(1,0),(1,1),(1,7),(2,2),(2,3),(2,8),(5,4),(5,5),
            (5,11),(10,4),(10,10),(10,11),(11,5),(11,10),(11,11),(16,16),(16,17),
            (16,22),(17,16),(17,17),(17,23),(24,18),(24,24),(24,25),(25,24),
            (25,25),(25,26),(26,25),(26,26),(26,27),(29,23),(29,29),(29,28)]

    halfs = [(27,26)]
    
    quarters = [(3,2),(3,3),(3,9),(3,4),(4,3),(4,10),(4,5),(4,4),(6,0),(6,7),
            (6,12),(6,6),(7,1),(7,6),(7,13),(7,7),(8,2),(8,9),(8,4),(8,8),
            (9,3),(9,8),(9,15),(9,9),(12,6),(12,13),(12,18),(12,12),(13,7),
            (13,12),(13,19),(13,13),(14,8),(14,15),(14,20),(14,14),(15,9),
            (15,14),(15,21),(15,15),(18,12),(18,19),(18,24),(18,18),(19,18),
            (19,13),(19,20),(19,19),(20,14),(20,19),(20,21),(20,20),(21,15),
            (21,20),(21,22),(21,21),(22,16),(22,21),(22,23),(22,22),(23,17),
            (23,22),(23,29),(23,23)]

    absorbing = [(28,28)]
            
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if (i,j) in thirds:
                p[i,j] = 1/3
            if (i,j) in halfs:
                p[i,j] = 1/2
            if (i,j) in quarters:
                p[i,j] = 1/4
            if (i,j) in absorbing:
                p[i,j] = 1
    return p

def get_minoutaur_transitions():
    m = np.zeros((30,30)) 
    thirds = [(1,0),(1,2),(1,7),(2,1),(2,3),(2,8),(3,2),(3,9),(3,4),(4,3),(4,10),
            (4,5),(6,0),(6,7),(6,12),(12,6),(12,13),(12,18),(18,12),(18,19),
            (18,24),(11,5),(11,10),(11,17),(17,11),(17,16),(17,23),(23,17),
            (23,22),(23,29),(25,19),(25,24),(25,26),(26,20),(26,25),(26,27),
            (27,21),(27,26),(27,28),(28,22),(28,27),(28,29)]

    mid = [7,8,9,10,13,14,15,16,19,20,21,22]
    
    halfs = [(0,1),(0,6),(5,4),(5,11),(24,18),(24,25),(29,23),(29,28)]
    
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if (i,j) in thirds:
                m[i,j] = 1/3

            if (i,j) in halfs:
                m[i,j] = 1/2

            if i in mid:
                m[i+1,i] = 1/4
                m[i-1,i] = 1/4
                m[i,i+6] = 1/4
                m[i,i-6] = 1/4

    return m

def get_transitions():
    p = get_player_transitions()
    m = get_minoutaur_transitions()
    plt.figure()
    plt.pcolormesh(p)
    plt.colorbar()
    plt.figure()
    plt.pcolormesh(m)
    plt.colorbar()
    plt.show()
    T = np.zeros((900,900))
    for i in range(30):
        for j in range(30):
            T[ i*30:(i+1)*30, j*30:(j+1)*30 ] = p * m[i,j]

    return T


def main():
    T = get_transitions()
    #print(T)
    pass


if __name__ == '__main__':
    main()
