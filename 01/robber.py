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

def q_learning(Q, state, a):
    """
    The main q_learning algorithm taken from p.107 from Introduction to Reinforcement Learning
    by Sutton.

    Input:
    @Q - numpy ndarray [S x A] representing the state-action matrix that stores the best
         action at state s
    @s - double representing the current state we are investigating
    @a - double, representing the current action we are doing at state s

    Output:
    @Q - numpy ndarray [S x A] updated state-action matrix.
    """
    q = Q[s, a]
    # We need the next state obtained by performing action a
    # Not yet implemented
    # s_next = step()
    # Place holder values until the reward and step function is complete
    s_next = s
    r = 1
    step_size = 0.99
    # TODO (Robert): Literature says the max of functions rather than the Q[s_next, action]
    Q[s, a] = Q[s, a] + step_size * (r + gamma * Q[s_next, action] - Q[s, action])


def step():
    """
    Function to take one step forward in the environment given the environment, state
    and action. Returns a new state
    """
    raise NotImplementedError("Error: Not yet implemented!")


def main():
    # Possible actions of the player
    # TODO (Robert): Can the police move?
    #          U, D, L, R, S
    actions = [0, 1, 2, 3, 4]
    # The map of the environment
    grid = np.ndarray([ROWS, COLS])
    # The total number of states.
    # The terminal state is when the robber gets caught(?)
    num_states = (ROWS * COLS)**2
    # The state action matrix. This is the Q function, that
    # is to be evaluated in the Q-learning algorithm
    Q = np.zeros([num_states, len(actions)])
    # Rewards. Should be
    R = np.zeros([num_states, len(actions)])
    # Update the rewards according to the specification.
    # TODO (Robert): Update the rewardsa w.r.t the states
    # of the grid. As in Problem 1 we have to consider
    # all the possible states that the board can have.
    # This is because we need to consider the future states
    # that the police might take.

    # Discount factor
    gamma = 0.8
    return


if __name__ == '__main__':
    ROWS = 4
    COLS = 4
    main()
