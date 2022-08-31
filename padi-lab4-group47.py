#### Activity 1.        


import numpy as np

def sample_transition(mdp, s, a):
    trans = []
    trans.append(s)
    trans.append(a)
    trans.append(mdp[3][s][a])
    
    trans.append(np.random.choice(len(mdp[2][a][s]), p=mdp[2][a][s]))
    
    return tuple(trans)


# #### Activity 2.        


import numpy as np
import numpy.random as rnd

def egreedy(Q, eps = 0.10):

    rand = np.random.random() 
    if rand < eps:
        j = np.random.randint(len(Q))
        return j
    else:
        policy = np.isclose(Q, Q.min())
        policy = policy/policy.sum()
    
        return np.random.choice(len(Q), p = policy)

 
# #### Activity 3. 


def mb_learning(mdp, n, qinit, Pinit, cinit):
    q = qinit
    c = cinit
    P = Pinit
    
    N = np.zeros((len(mdp[0]), len(mdp[1])))
    
    s = np.random.choice(len(mdp[0]))
    
    retTuple = []
    
    for i in range(n):
        a = egreedy(q[s],0.15)
        s_tmp = sample_transition(mdp, s, a)
        
        N[s][a] = N[s][a] + 1
        
        step = 1/(N[s][a] + 1)
        
        c[s][a] = c[s][a] + step*(s_tmp[2] - c[s][a])
        
        I = np.zeros((len(mdp[0])))
        I[s_tmp[3]] = 1
        
        P[a][s, :] = P[a][s, :] + step*(I - P[a][s, :])
        
        q[s][a] = c[s][a] + mdp[4] * P[a][s, :].dot(q.min(axis = 1, keepdims = True))
        
        s = s_tmp[3]
        
        
        
    retTuple.append(q)
    retTuple.append(P)
    retTuple.append(c)
    
    return tuple(retTuple)


# #### Activity 4. 



def qlearning(mdp, n, qinit):
    gamma = mdp[4]
    alpha = 0.3
    eps = 0.15
    
    q = qinit
    state = np.random.choice(len(mdp[0]))

    for i in range(n):
        action = egreedy(q[state], eps)

        state, action, cost, next_state = sample_transition(mdp, state, action)
        
        q[state][action] = q[state][action] + alpha *(cost + gamma *  q[next_state].min() - q[state][action])
        state = next_state    
    return q



# #### Activity 5. 


def sarsa(mdp, n, qinit):
    gamma = mdp[4]
    alpha = 0.3
    eps = 0.15
    
    q = qinit
    state = np.random.choice(len(mdp[0]))
    action = egreedy(q[state], eps)

    for i in range(n):
        state, action, cost, next_state = sample_transition(mdp, state, action)
        actionNext = egreedy(q[next_state], eps)
        
        q[state][action] = q[state][action] + alpha *(cost + gamma * q[next_state][actionNext] - q[state][action])
        state = next_state
        action = actionNext
        
    return q


# #### Activity 6.
#     We see that the results from using the Q-Learning and SARSA methods have very similar performances.
#     
#     The error in the model based learning approach is better than the two previously mentioned. It has a lower error with the same number of iterations as Q-learning and SARSA. 
#     
#     With a few number of iterations the error is quite similar with every method, but as the number of iterations increase the model based learning method is improving more than the other methods.  
