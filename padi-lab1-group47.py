
 


# #### Activity 1

import numpy as np

def load_chain(filePath):
    b = np.load(filePath)
    return (tuple(str(i) for i in range(1,len(b[0]) + 1)), b)


# #### Activity 2


def prob_trajectory(markov_chain, trajectory):
    totalProb = 1
    for i in range(0, len(trajectory) - 1):
        totalProb = totalProb * markov_chain[1][int(trajectory[i]) - 1][int(trajectory[i+1]) - 1]
    return totalProb

# #### Activity 3


import numpy as np

def stationary_dist(markov_chain):   
    evals, evecs = np.linalg.eig(markov_chain[1].T)
    evec = evecs[:,np.isclose(evals, 1)]
    evec = evec[:, 0]
    stationary = evec/ evec.sum()
    stationary = stationary.real
    return stationary



# #### Activity 4.



def compute_dist(markov_chain, row_vector, N):
    return np.matmul(row_vector, np.linalg.matrix_power(markov_chain[1], N))




# #### The chain is ergodic, because it is possible to go from every state to any other state,  in a certain amount of moves.


# #### Activity 5



def simulate(markov_chain, row_vector, N):
    b = np.random.choice(len(markov_chain[1] + 1), 1, p=row_vector[0])
    traj = [str(b[0] + 1)]
    
    for i in range (0, N):
        y = np.random.choice(len(markov_chain[1][0]), 1, p = markov_chain[1][b][0])
        b = y
        traj.append(str(y[0] + 1))
    return tuple(traj)

