import numpy as np

N_COLS = 6
N_ROWS = 5
N_STATES = N_COLS * N_ROWS
ACTIONS = ["R","L","U","D","S"]
N_ACTIONS = len(ACTIONS)
walls = [(1,2),(2,1),(7,8),(8,7),(13,14),(14, 13),(9,10),(10,9),(15,16),(16,15),(10,16),(16,10),(11,17),(17, 11),(19,25),(25,19),(20,26),(26,20),(21,27),
        (27,21),(27,28),(28,27),(22,28),(28,22)]

def move(row,col, action):
    current_state = row * N_COLS + col
    s_prims = np.zeros((N_ROWS*N_COLS,))
    if action == "R":
        col = min(N_COLS-1, col+1)

    if action == "L":
        col = max(0, col-1)
    
    if action == "U":
        row = max(0, row-1)

    if action == "D":
        row = min(N_ROWS-1, row+1)

    if action == "S":
        s_prims[current_state] = 1
    
    new_state = row * N_COLS + col
    if not (current_state, new_state) in walls:
        s_prims[new_state] = 1
    
    # CHECK HOW TO MARK GOAL STATE
    #elif new_state == 28:
    #    s_prims

    else:
        s_prims[new_state] = 0

    return s_prims

def create_T():
    T = np.zeros((N_STATES,N_STATES,N_ACTIONS))
    counter = 0
    for row in range(N_ROWS):
        for col in range(N_COLS):
            T[counter,:,0] = move(row,col,ACTIONS[0])
            T[counter,:,1] = move(row,col,ACTIONS[1])
            T[counter,:,2] = move(row,col,ACTIONS[2])
            T[counter,:,3] = move(row,col,ACTIONS[3])
            T[counter,:,4] = move(row,col,ACTIONS[4])
            counter += 1
    return T


T = create_T()
np.save('T.npy',T)
#print(T[0,:,:])


