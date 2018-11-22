import numpy as np
N_ROWS = 5
N_COLS = 6
N_STATES = N_ROWS*N_COLS 
GOAL = 28
ACTIONS = ["S", "U", "D", "R" , "L"]
M_ACTIONS = {"U": np.array([0, 1]), "D": np.array([0, -1]), "R": np.array([1, 0]), "L": np.array([-1, 0])}
# movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]
T = 15
N_ACTIONS = len(ACTIONS)

walls = [(1,2),(2,1),(7,8),(8,7),(13,14),(14, 13),(9,10),(10,9),(15,16),(16,15),(10,16),(16,10),(11,17),(17, 11),(19,25),(25,19),(20,26),(26,20),(21,27),
        (27,21),(27,28),(28,27),(22,28),(28,22)]


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
    old_row = row
    old_col = col
    prev_state = row * N_COLS + col
    trans_prob = 0.2

    if action == "S":
        state = trans_prob, row * N_COLS + col
    elif action == "R":
        col = min(N_COLS-1, col+1) 
    elif action == "L":
        col = max(0, col-1)
    elif action == "U":
        row = max(0,row-1)
    elif action == "D":
        row = min(N_ROWS-1, row+1)

    state = row * N_COLS + col
    reward = 1 if  state == GOAL else 0
    if (prev_state, state) in walls:
        state = old_row * N_COLS + old_col
        trans_prob = 0.2
        reward = 0



    return reward, trans_prob, state



def build(m_path):
    # print(m_path)
    # minotaur_coord = [None] * len(m_path)
    # minotaur_walk = [None] * len(m_path)
    # minotaur_walk[0] = 4 * N_COLS + 4
    # minotaur_coord[0] = [4, 4]
    # for i in range(1, len(m_path)):
    #     minotaur_walk[i] = move(minotaur_coord[i][0], minotaur_walk) # minotaur_walk[i-1] + M_ACTIONS[m_path[i]]
    # print(minotaur_walk)
    P = {state : {action: [] for action in ACTIONS} for state in range(N_STATES)}
    for row in range(N_ROWS):
        for col in range(N_COLS):
            state = row * N_COLS + col
            for action in ACTIONS:
                l = P[state][action]
                reward, trans_prob, s_prim = move(row, col, action)
                # print(f"{state} -> {s_prim}, {action}: {reward}")
                l.append((trans_prob, s_prim, reward, is_goal(s_prim)))

    return P

def policy_iteration(P, policy):
    iter = 0
    old_policy = policy.copy()
    while iter < 15:
        print("\n iter: {}".format(iter))
        V = compute_policy(P,policy) 
        policy = update_policy(V, P, policy)
        old_policy = policy
        if np.all(policy == old_policy):
            break
        iter +=1
    return policy, V

def compute_policy(P,policy):
    diff = 99999
    eps = 0.01
    V = np.zeros((N_STATES,))
    x = 0
    while True:
        delta = 0
        #V_current = V.copy()
        for state in range(N_STATES):
            v = 0
            #action = policy[state]
            for action in ACTIONS:
                trans_prob, s_prim, reward, _  = P[state][action][0]
                # print(trans_prob, s_prim, reward)
                v += trans_prob * (reward + V[s_prim])
            # _ = help_printer(V)
            delta = max(delta, np.abs(v - V[state]))
            print(delta)
            V[state] = v
        if delta < eps:
            break
        x+=1
    return V

def update_policy(V, P, policy):
    # policy = list(np.arange(N_STATES))
    pi = np.zeros((N_STATES,))
    unchanged = True
    while unchanged:
        for state in range(N_STATES):
            # pi = 0
            # q = np.zeros((N_ACTIONS,))
            value = -10
            best_action = ""
            for a, action in enumerate(ACTIONS):
                trans_prob, s_prim, _, _ = P[state][action][0]
                q = trans_prob * V[s_prim]
                if q > value:
                    value = q
                    best_action = action
            # print(policy[state])
            p, s_new, _, _ = P[state][policy[state]][0]
            if value > p * V[s_new]:
                print("Changing the policy.")
                policy[state] = best_action
                unchanged = False
                    # r += p * V[s_new]
            # q[a] = sum([trans_prob * (reward + V[s_prim]) for trans_prob, s_prim, reward, _ in P[state][action]])
        # PI[state] = pi
        # idx = np.argmax(PI)
        # policy[state] = ACTIONS[idx]
    print("Done!")
    return policy


minotaur_path = np.random.choice(["U", "D", "R", "L"], size=T)

P = build(minotaur_path)
policy = np.random.choice(ACTIONS, size=N_STATES)
new_policy, V = policy_iteration(P,policy)
import matplotlib.pyplot as plt
print(policy)
# V_mat = help_printer(V)[::-1]
# norm1 = V_mat / np.linalg.norm(V_mat)
# # print(walls)
# plt.pcolormesh(norm1)
# plt.colorbar()
# plt.show()
