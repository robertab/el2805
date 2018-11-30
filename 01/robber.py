#!/usr/bin/python3

"""
The environment for Problem 3: Bank Robbing (Reloaded)

A  .  .  .  .

.  B  .  .  .

.  .  .  .  .

.  .  .  .  .

The player (us) starts at (0, 0) and the police starts at (4, 4)
The reward for each round (staying or moving?) in the bank results in a reward of +1
The reward for getting caught by the police results in a reward of -10
(Same cell as the police)

Actions are performed uniformly at random (up, down, left, right, stay).
We observer the position of the robber and the police at each time step.

a) Solve the problem by implementing the Q-learning algorithm exploring actions uniformly at
   random. Create a plot of the value function over time (in particular, for the initial state),
   showing the convergence of the algorithm. Note: Expect the value function to converge
   after roughly 10 000 000 iterations (for step size 1/n(s, a) 2/3 , where n(s, a) is the number of
   updates of Q(s, a))
"""

import numpy as np
import doctest
import matplotlib.pyplot as plt
from utils import *

def q_learning(Q, state, q_values, actions, R, num):
    """
    The main q_learning algorithm taken from p.107 from Introduction to Reinforcement Learning
    by Sutton.

    Input:
    @Q - numpy ndarray [S x A] representing the state-action matrix that stores the best
         action at state s
    @s - double representing the current state we are investigating

    Output:
    @Q - numpy ndarray [S x A] updated state-action matrix.
    """
    gamma = 0.8
    # Random action of the robber
    action = np.random.randint(len(actions))
    r, new_state = R[state][action]
    alpha = 1 / ((num[state, action] + 1)**(2/3))
    Q[state, action] += alpha * (r + gamma * np.max(Q[new_state, :]) - Q[state, action])
    # Check the initial state and store for later use
    q_values.append(np.max(Q[0, :]))
    num[state, action] += 1
    return Q, state, q_values, num
    # q = Q[s, a]
    # # We need the next state obtained by performing action a
    # # Not yet implemented
    # # s_next = step()
    # # Place holder values until the reward and step function is complete
    # s_next = s
    # r = 1
    # step_size = 0.99
    # # TODO (Robert): Literature says the max of functions rather than the Q[s_next, action]
    # Q[s, a] = Q[s, a] + step_size * (r + gamma * Q[s_next, action] - Q[s, action])


def step(row, col, a):
    """
    Function to take one step forward in the environment given the environment, state
    and action. Returns a new state. This also checks if the new position is inside
    the grid.
    """
    if a == 4: # Stand still
        return [row, col]
    if a == 0:
        row = max(0, north([row, col])[0])
    if a == 1:
        row = min(ROWS-1, south([row, col])[0])
    if a == 2:
        col = max(0, left([row, col])[1])
    if a == 3:
        col = min(COLS-1, right([row, col])[1])
    return [row, col]


def rewards(R):
    n_states, n_actions = 256, 5
    for s in range(n_states):
        R[s] = {a: [] for a in range(5)}
        r = 0
        # Get the position for the state
        pos = np.unravel_index(s, (ROWS, COLS, ROWS, COLS))
        pos_robber, pos_police = np.array(pos[0:2]), np.array(pos[2:4])
        # Calculate the possible actions that the robber can perform
        for a in range(n_actions):
            new_pos_robber = step(pos_robber[0], pos_robber[1], a)
            # The police moves randomly on the grid
            new_pos_police = step(pos_police[0], pos_police[1], np.random.randint(4))
            # print(new_pos_police)
            new_pos = new_pos_robber + new_pos_police
            new_s = np.ravel_multi_index(new_pos, (ROWS, COLS, ROWS, COLS))
            # Was caught by the police
            if np.all(new_pos_robber == new_pos_police):
                R[s][a] = [-10, new_s]
            # Entered the bank
            if new_pos_robber == [1, 1]:
                R[s][a] = [1, new_s]
            else:
                R[s][a] = [0, new_s]
    return R

def epsilon_greedy(Q, state, actions, epsilon=0.1):
    num_actions = len(actions)
    a = np.ones(num_actions) * (epsilon / num_actions)
    action = np.argmax(Q[state, :])
    a[action] = a[action] + (1 - epsilon)
    return a


def main():
    # Possible actions of the player
    # TODO (Robert): Can the police move?
    #          U,          D,      L,       R,      S
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    # The map of the environment
    grid = np.ndarray([ROWS, COLS])
    # The total number of states.
    # The terminal state is when the robber gets caught(?)
    num_states = (ROWS * COLS)**2
    # The state action matrix. This is the Q function, that
    # is to be evaluated in the Q-learning algorithm
    Q = np.zeros([num_states, len(actions)])
    num = np.zeros([num_states, len(actions)])
    # Rewards. Should be
    R = {}
    # np.zeros([num_states, len(actions)])
    # Discount factor
    gamma = 0.8
    # Calculate the rewards based on the police movements etc.
    R = rewards(R)
    state = np.ravel_multi_index((0, 0, 3, 3), (ROWS, COLS, ROWS, COLS))
    # pos = (0, 0, 3, 3)
    iter = 1000000
    q_values = []
    entered = 0
    # Q-learning
    # for i in range(iter):
    #     # Q, new_state, q_values, num = q_learning(Q, state, q_values, actions, R, num)
    #     # Random action of the robber
    #     action = np.random.randint(len(actions))
    #     r, new_state = R[state][action]
    #     alpha = 1 / ((num[state, action] + 1)**(2/3))
    #     Q[state, action] += alpha * (r + gamma * np.max(Q[new_state, :]) - Q[state, action])
    #     # Check the initial state and store for later use
    #     q_values.append(np.max(Q[0, :]))
    #     num[state, action] += 1
    #     state = new_state
    # SARSA with epsilon-greedy
    epsilon = 0.4
    a = epsilon_greedy(Q, state, actions, epsilon)
    action = np.random.choice(np.arange(len(a)), p=a)
    for i in range(iter):
        # Q, new_state, q_values, num = q_learning(Q, state, q_values, actions, R, num)
        # Random action of the robber
        r, new_state = R[state][action]
        # action = np.random.randint(len(actions))
        new_a = epsilon_greedy(Q, new_state, actions, epsilon)
        new_action = np.random.choice(np.arange(len(new_a)), p=new_a)
        alpha = 1 / ((num[state, action] + 1)**(2/3))
        Q[state, action] += alpha * (r + gamma * Q[new_state, new_action] - Q[state, action])
        # Check the initial state and store for later use
        q_values.append(np.max(Q[0, :]))
        num[state, action] += 1
        state = new_state
        action = new_action


    # plt.legend()
    plt.plot(q_values, label="q(0) - value")
    plt.xlabel("Iterations")
    plt.title("SARSA using epsilon-greedy exploration")
    plt.ylabel("State-action value @ state 0 (initial)")
    plt.legend()
    plt.show()

    # Update the rewards according to the specification.
    # TODO (Robert): Update the rewardsa w.r.t the states
    # of the grid. As in Problem 1 we have to consider
    # all the possible states that the board can have.
    # This is because we need to consider the future states
    # that the police might take.
    # Discount factor

    return


if __name__ == '__main__':
    doctest.testmod()
    ROWS = 4
    COLS = 4
    main()
