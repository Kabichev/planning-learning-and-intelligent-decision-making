
#Activity 1.        



import numpy as np


def load_pomdp(filename, gamma):
    
    b = np.load(filename)
    
    POMDP_tuple = []
    POMDP_tuple.append((b['X']))
    POMDP_tuple.append(b['A'])
    POMDP_tuple.append((b['Z']))
    
    list_P = []
    for i in b['P']:
        list_P.append(i)
    list_P = tuple(list_P)
    POMDP_tuple.append(list_P)
    
    
    list_O = []
    for i in b['O']:
        list_O.append(i)
    list_O = tuple(list_O)
    POMDP_tuple.append(list_O)

    
    POMDP_tuple.append(b['c'])
    POMDP_tuple.append(float(gamma))
    
    POMDP_tuple_done = tuple(POMDP_tuple)


    return POMDP_tuple_done




# #### Activity 2.

import numpy.random as rand

def gen_trajectory(pomdp, x0, n):
    rand.seed(42)
    generated_traj = []
    xState = x0
    traj = np.zeros(n+1, dtype = int)
    actions = np.zeros(n, dtype = int)
    obsv = np.zeros(n, dtype = int)
    traj[0] = int(x0)
    for i in range(n):
        a = rand.choice(len(pomdp[1]))
        x_next = rand.choice(len(pomdp[0]), p = pomdp[3][a][xState])
        obs = rand.choice(len(pomdp[2]), p = pomdp[4][a][x_next])
        xState = x_next
        traj[i+1] = int(x_next)
        actions[i] = int(a)
        obsv[i] = int(obs)
    generated_traj.append(traj)
    generated_traj.append(actions)
    generated_traj.append(obsv)
    return tuple(generated_traj)

#Activity 3


import numpy as np

import numpy.random as rand

def sample_beliefs(pomdp, n):
    x0 = rand.randint(len(M[0]))
    belief_states = []
    a = gen_trajectory(pomdp, x0, n)
    belief_prev = np.ones((1, len(pomdp[0])))/len(pomdp[0])
    belief_states.append(belief_prev)
    join = True
    for i in range(0, n):
        belief_prev = belief_update(pomdp, belief_prev, a[1][i], a[2][i])
        belief_states.append(belief_prev)
    for i in range(len(belief_states)):
        
        a = np.linalg.norm(belief_prev - belief_states[i])
        
        if (a < 0.001):
            join = False
            
    if join == True:
        belief_states.append(belief)

    return belief_states

def belief_update(pomdp, belief, action, observation):
    updated_belief = belief.dot(pomdp[3][action] * pomdp[4][action][:, observation])
    return updated_belief/updated_belief.sum()





# Activity 4


import numpy as np

def solve_mdp(POMDP):
    error = 1
    Q_pre = np.zeros((len(POMDP[0]), len(POMDP[1])))
    Q = np.zeros((len(POMDP[0]), len(POMDP[1])))
    
    while (error > 1e-8):
        for i in range(len(POMDP[1])):          
            Q[:,i:i+1] = POMDP[5][:, i: i+1] + POMDP[3][i].dot(Q_pre.min(axis = 1, keepdims = True)) * POMDP[6]
        error = np.linalg.norm(Q - Q_pre) 
        Q_pre = Q.copy()
    return Q
    



# #### Activity 6
# 