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
"""

import numpy as np



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
