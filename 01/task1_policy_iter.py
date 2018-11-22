import numpy as np
N_ROWS = 5
N_COLS = 6
N_STATES = N_ROWS*N_COLS 
GOAL = 28
ACTIONS = ["S", "R", "L", "U" , "D"]
N_ACTIONS = len(ACTIONS)

walls = np.array([(2,3),(3,2),(8,9),(9,8),(14,15),(15,14),(10,11),(11,10),(16,17),(17,16),(11,17),(17,11),(12,18),(18,12),(20,26),(26,20),(21,27),(27,21),(22,28),
        (28,22),(28,29),(29,28),(23,29),(29,23)]) - 1


def help_printer(input):
    V_mat = np.zeros((N_ROWS,N_COLS))
    for row in range(N_ROWS):
        for col in range(N_COLS):
            print(input[row*N_COLS+col], end=' ')
            V_mat[row,col] = input[row*N_COLS+col]
        print()
    return V_mat

def is_goal(state):
    return state == GOAL

def move(row, col, action):
    
    trans_prob = 0.2
    
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

    if (row,col) in walls:
        return 0, row * N_COLS + col

    return trans_prob, row * N_COLS + col
    

def build():
    P = {state : {action: [] for action in ACTIONS} for state in range(N_STATES)}
    for row in range(N_ROWS):
        for col in range(N_COLS):
            state = row * N_COLS + col
            for action in ACTIONS:
                l = P[state][action]
                trans_prob, s_prim = move(row, col, action)
                reward = 1 if s_prim == GOAL else 0
                l.append((trans_prob, s_prim, reward, is_goal(s_prim)))
    print(P)
    return P

def policy_iteration(P,policy):
    iter = 0
    while iter < 10:
        print("\n iter: {}".format(iter))
        V = compute_policy(P,policy)
        policy = update_policy(V, P)
        iter +=1
    return policy, V

def compute_policy(P,policy):
    diff = 99999
    eps = 0.1
    V = np.zeros((N_STATES,))
    x = 0
    while True:
        #eps = 0
        delta = 0
        #V_current = V.copy()
        for state in range(N_STATES):
            v = 0
            #action = policy[state]
            for action in ACTIONS:
                trans_prob, s_prim, reward, _  = P[state][action][0]
                v += trans_prob * (reward + V[s_prim])
                    #V[state] = sum([trans_prob * (reward + V_current[s_prim]) for trans_prob, s_prim, reward, _  in P[state][action]])
                print(state,s_prim, reward, action)
            _ = help_printer(V)
            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
        if delta < eps:
            break
        x+=1
        #print(policy)
        #print(f'Current: {V_current}')
        #print(f'New: {V}')
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
new_policy, V = policy_iteration(P,policy)
import matplotlib.pyplot as plt
V_mat = help_printer(V)[::-1]
print(walls)
plt.pcolormesh(V_mat)
plt.show()
