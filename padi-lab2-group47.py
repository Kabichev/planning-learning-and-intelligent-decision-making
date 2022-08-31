#Activity 1

import numpy as np 


def load_mdp(filename, g):
    b = np.load(filename)
    mdp_tuple = []
    mdp_tuple.append((b['X']))
    mdp_tuple.append(b['A'])
    
    list_P = []
    for i in b['P']:
        list_P.append(i)
    mdp_tuple.append(list_P)
    mdp_tuple.append(b['c'])
    mdp_tuple.append(float(g))
    

    mdp_tuple_done = tuple(mdp_tuple)

    return mdp_tuple_done






# #### Activity 2.

import numpy as np
def noisy_policy(mdp_tuple, a, eps):
    policy = np.zeros((len(mdp_tuple[0]), len(mdp_tuple[1])))
    for i in range(len(mdp_tuple[0])):
        for j in range(len(mdp_tuple[1])):
            if j == a:
                policy[i][j] = 1-eps
            else:
                policy[i][j] = eps/(len(mdp_tuple[1])-1)
    return policy
    

#Activity 3

def evaluate_pol(mdp_tuple, policy):
    cost_to_go = np.zeros(len(mdp_tuple))
    P = np.zeros((len(mdp_tuple[0]), len(mdp_tuple[0])))
    for i in range(len(mdp_tuple[1])):
        P += policy[:, i: i+1] * mdp_tuple[2][i]
    C = policy * mdp_tuple[3]
    C = C.sum(axis = 1, keepdims = True)
    cost_to_go = np.linalg.inv(np.identity(len(P)) - mdp_tuple[4]*P).dot(C)        
            
    return cost_to_go

#Activity 4


import time
def value_iteration(mdp_tuple):
    start = time.time()
    
    #variables:
    opt_cost_to_go = np.zeros((len(mdp_tuple[0]), 1))
    num_iter = 0
    error = 1
    while (error > 1e-8):
        Q = []  
        for i in range(len(mdp_tuple[1])):          
            Q.append(mdp_tuple[3][:, i: i+1] + mdp_tuple[2][i].dot(opt_cost_to_go) * mdp_tuple[4])          
        opt_cost_to_go_new = np.min(Q, axis = 0)   
        error = np.linalg.norm(opt_cost_to_go_new - opt_cost_to_go)  
        num_iter += 1      
        opt_cost_to_go = opt_cost_to_go_new
    end = time.time()
    print("Execution time: " + str((end - start)) + " seconds")
    print("N. iterations: " + str(num_iter))   
    return opt_cost_to_go

#Activity 5


import time

def policy_iteration(mdp_tuple):
    start = time.time()
    pi = np.ones((len(mdp_tuple[0]), len(mdp_tuple[1])))/ len(mdp_tuple[1])
    quit = False
    num_iter = 0
    

    
    while not quit:        
        J = evaluate_pol(mdp_tuple, pi) 
        Q = []            
        for i in range(len(mdp_tuple[1])):          
            Q.append(mdp_tuple[3][:, i: i+1] + mdp_tuple[2][i].dot(J) * mdp_tuple[4]) 
        
        pinew = np.zeros((len(mdp_tuple[0]), len(mdp_tuple[1])))
        
        for i in range(len(mdp_tuple[1])):
            pinew[:, i, None] = np.isclose(Q[i], np.min(Q, axis = 0), atol = 1e-8, rtol = 1e-8).astype(int)
            
        pinew = pinew / np.sum(pinew, axis=1, keepdims = True)
        
        quit = (pi == pinew).all()
        
        pi = pinew
        num_iter += i


    end = time.time()
    print("Execution time: " + str((end - start)) + " seconds")
    print("N. iterations: " + str(num_iter))

    return pi

#Activity 6

#Is not started