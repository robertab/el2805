import numpy as np
N_ROWS = 6
N_COLS = 5
N_STATES = N_ROWS*N_COLS 
N_ACTIONS = 5
GOAL = 28
ACTIONS = ["S", "R", "L", "U" , "D"]


def is_goal(state):
    return state == GOAL

def move(row, col, action):
    
    trans_prob = 1

    if action == "S":
        return trans_prob, row * N_COLS + col
    elif action == "R":
        col = min(N_COLS-1, col+1) 
    elif action == "L":
        col = max(0, col-1)
    elif action == "U":
        row = max(0,row-1)
    elif action == "D":
        row = min(N_ROWS-1, row+1)

    return trans_prob, row * N_COLS + col
    

def build():
    P = {state : {action: [] for action in ACTIONS} for state in range(N_STATES)}
    for row in range(N_ROWS):
        for col in range(N_COLS):
            for action in ACTIONS:
                l = P[row][action]
                reward = 1 if row * N_COLS + col == GOAL else 0
                trans_prob, s_prim = move(row, col, action)
                l.append((trans_prob, s_prim, reward, is_goal(s_prim)))
    return P

def policy_iteration(P,policy):
    iter = 0
    while iter < 1:
        V = compute_policy(P,policy)
        policy = update_policy(V, P)
        iter +=1
    return

def compute_policy(P,policy):
    diff = 99999
    eps = 0.1
    V = np.zeros((N_STATES,))
    while eps < diff:
        V_new = V.copy()
        for state in range(N_STATES):
            action = policy[state]
            #print(action)
            # summing over all actions
            V_new[state] = sum([trans_prob * (reward + V[s_prim]) for (trans_prob, s_prim, reward, _ ) in P[state][action]])
        diff = sum(np.fabs(V - V_new))
        print(diff)
    return V_new

def update_policy(policy, P):

    return policy

P = build()
policy = np.random.choice(ACTIONS, size=N_STATES)
print(P)
print(policy)
#policy_iteration(P,policy)

