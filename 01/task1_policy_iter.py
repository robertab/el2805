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
    while iter < 9:
        V = compute_policy(P,policy)
        policy = update_policy(V, P)
        iter +=1
        print("\n iter: {}".format(iter))
    return policy

def compute_policy(P,policy):
    diff = 99999
    eps = 0.0001
    V = np.zeros((N_STATES,))
    x = 0
    while eps < diff:
        V_prev = V.copy()
        for state in range(N_STATES):
            action = policy[state]
            # summing over all actions to get to some next state.
            V[state] = sum([trans_prob * (reward + V_prev[s_prim]) for trans_prob, s_prim, reward, _  in P[state][action]])
        diff = np.sum(np.fabs(V_prev - V))
        #print(V_new)
        print(diff)
        print("compute policy: {}".format(x))
        x+=1
    return V

def update_policy(V, P):
    policy = list(np.arange(N_STATES))
    for state in range(N_STATES):
        q = np.zeros((N_ACTIONS,))
        for a, action in enumerate(ACTIONS):
            q[a] = sum([trans_prob * (reward + V[s_prim]) for trans_prob, s_prim, reward, _ in P[state][action]])
        idx = np.argmax(q)
        policy[state] = ACTIONS[idx]
    return policy

P = build()
policy = np.random.choice(ACTIONS, size=N_STATES)
print(policy)
new_policy = policy_iteration(P,policy)
print(policy)
print(new_policy)

