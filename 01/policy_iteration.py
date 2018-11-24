import numpy as np
from typing import List
from numpy import ndarray as Tensor
import matplotlib.pyplot as plt
from pprint import pprint

N_COLS = 6
N_ROWS = 5
N_STATES = N_COLS * N_ROWS
ACTIONS = ["R","L","U","D","S"]
N_ACTIONS = len(ACTIONS)
walls = [(1,2),(2,1),(7,8),(8,7),(13,14),(14, 13),(9,10),(10,9),(15,16),(16,15),(10,16),(16,10),(11,17),(17, 11),(19,25),(25,19),(20,26),(26,20),(21,27),
        (27,21),(27,28),(28,27),(22,28),(28,22)]

class MDP:
    def __init__(self, states: List, actions: List, r: List, T: Tensor, gamma: float) -> None:
        self.states = states
        self.actions = actions
        self.T = T
        self.r = r
        self.gamma = gamma

def move_minotaur(row,col):
    actions = []
    if not col == min(N_COLS-1, col+1): # "R"
        actions.append('R')

    if not col == max(0, col-1):
        actions.append('L')

    if not row == max(0, row-1):
        actions.append('U')

    if not row == min(N_ROWS-1, row+1):
        actions.append('D')

    action = np.random.choice(actions, size=1)

    current_state = row * N_COLS + col
    
    if action == "R":
        col = min(N_COLS-1, col+1)

    if action == "L":
        col = max(0, col-1)
    
    if action == "U":
        row = max(0, row-1)

    if action == "D":
        row = min(N_ROWS-1, row+1)

    #if action == "S":
        #s_prims[current_state] = 1
    
    #new_state = row * N_COLS + col

    return row,col


def move_agent(row: int, col: int, action: str) -> (int, int):
    prev_row = row
    prev_col = col
    prev_state = row*N_COLS+col
    if action == "R":
        col = min(N_COLS-1, col+1)

    if action == "L":
        col = max(0, col-1)

    if action == "U":
        row = max(0, row-1)

    if action == "D":
        row = min(N_ROWS-1, row+1)

    if action == 'S':
        pass

    new_state = row*N_COLS+col  
    
    if not (prev_state,new_state) in walls:
        return row,col
    else:
        return prev_row, prev_col


def format_utility(U):
    utility = np.zeros((N_ROWS,N_COLS))
    for row in range(N_ROWS):
        for col in range(N_COLS):
            utility[row,col] = U[row*N_COLS+col]
    return utility

def printer(input):
    for row in range(N_ROWS):
        for col in range(N_COLS):
            print(input[row*N_COLS+col], end=" ")
        print()
    print()


def select_best_action(state: int, U: List, mdp: MDP) -> int:
    possible_actions = np.zeros((N_ACTIONS,))
    for i,action in enumerate(mdp.actions):
        possible_actions[i] = np.sum(np.multiply(U , mdp.T[state,:,i]))
    return np.argmax(possible_actions)

def policy_evaluation(policy: List, U: List, mdp: MDP) -> List:
    for state in mdp.states:
        # policy gives an acton
        action = ACTIONS.index(policy[state])
        U[state] = mdp.r[state] + mdp.gamma * np.sum(np.multiply(U, mdp.T[state,:,action]))
    return U

def policy_iteration(mdp: MDP) -> List:
    policy = np.random.choice(ACTIONS, size=N_STATES)
    U = np.zeros((len(mdp.states),))
    iter = 0
    policy[28] = 'S'
    eps = 1e-4
    while iter < 20:
        U_prev = U.copy()
        U = policy_evaluation(policy, U, mdp)
        delta = np.abs(U-U_prev).max()
        #if delta < eps * (1-mdp.gamma) / mdp.gamma:
        #    break
        for state in mdp.states:
            action = select_best_action(state,U,mdp)
            if action != policy[state] and state != 28:
                policy[state] = ACTIONS[action]
        iter += 1
        #printer(policy)
    
    return policy, U, iter, delta, eps

def update_grid(a_row,a_col, m_row, m_col):
    grid = [['-']*N_COLS for i in range(N_ROWS)]
    grid[a_row][a_col] = 'A' 
    grid[m_row][m_col] = 'M'
    return grid

def main():
    # transition matrix for any given action. size=(N_STATES,N_STATES,N_ACTIONS)
    T = np.load('T.npy')
    gamma = 0.99
    # States
    states = list(range(N_STATES))
    # create reward vector
    r = np.zeros((N_STATES,)) 
    # mark terminal state
    terminal_state = 28
    r[terminal_state] = 1
    # mark terminal state in Transistion Matrix
    T[terminal_state,:,:] = 0
    
    #mdp = MDP(states, ACTIONS, r, T, gamma)
    
    m_state = 28
    m_state_prev = m_state
    #mdp.r[m_state] = -1
    #T[m_state_prev,:,:] = 0
    m_row = 4
    m_col = 4
    
    utils = []
    minotaur_pos = []
    TIME = 15
    u = np.zeros((N_STATES,TIME))
    p = np.random.choice(ACTIONS, size=(N_STATES,TIME))# [list(range(mdp.states)) for _ in range(T)]
    print(p.shape)
    print(u.shape)
    for t in range(TIME):
        mdp = MDP(states, ACTIONS, r.copy(), T, gamma)
        mdp.r[m_state] = -1
        
        policy, U, iter, delta, eps = policy_iteration(mdp)
        
        u[:,t] = U
        p[:,t] = policy
        
        minotaur_pos.append([m_row,m_col])
        m_row, m_col = move_minotaur(m_row, m_col)
        m_state = m_row*N_COLS+m_col
        
        #mdp.r[m_state_prev] = 0
        #mdp.r[m_state] = -1
        #if not m_state == 28:
        #    mdp.r[28] = 1

        utility = format_utility(U)

        print("============= RESULT =============")
        print()
        print("Iterations: " + str(iter))
        print("Delta: " + str(delta))
        print("Gamma: " + str(gamma))
        print("Epsilon: " + str(eps))
        print(f"Utility:\n {utility}")
        print(f"policy:")
        printer(policy)
        print(f'm_state: {m_state_prev}')
        print("==================================")
        
        m_state_prev = m_state

        #plt.pcolormesh(utility[::-1])
        #plt.show()


    actions = []
    a_row = 0
    a_col = 0
    print(f'  A POS  M POS')
    for i in range(TIME):
        print(i,[a_row,a_col],minotaur_pos[i])
        if minotaur_pos[i] == [a_row,a_col]:
            print(f"AGENT DIED after {i} iterations!!!")
        elif (a_row,a_col) == (4,4):
            print(f'AGENT WON!!!')
        
        m_row, m_col = minotaur_pos[i]
        grid = update_grid(a_row,a_col,m_row,m_col)
        pprint(grid)
        
        action = p[a_row*N_COLS+a_col,i]
        actions.append(action)
        a_row, a_col = move_agent(a_row,a_col,action)
        util = u[a_row*N_COLS+a_col,i]
        utils.append(util)
    

    print(utils)
    print(actions)
    plt.plot(utils)
    plt.show()

if __name__=="__main__":
    main()
