import numpy as np


# General function to expected hitting time for Exercise 2.1
def compute_Phi_ET(P, ns=100):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        ns {int} -- largest step to consider

    Returns:
        Phi_list {numpy.array} -- (ns + 1) x n x n, the Phi matrix for time 0, 1, ...,ns
        ET {numpy.array} -- n x n, expected hitting time approximated by ns steps ns
    '''

    # Add code here to compute following quantities:
    # Phi_list[m, i, j] = phi_{i,j}^{(m)} = Pr( T_{i, j} <= m )
    # ET[i, j] = E[ T_{i, j} ] ~ \sum_{m=1}^ns m Pr( T_{i, j} = m )
    # Notice in python the index starts from 0

    Phi_list = np.zeros([ns+1,P.shape[0],P.shape[1]])
    Phi_list[0] = P
    for i in range(1,ns+1):
        Phi_list[i] = np.matmul(P, Phi_list[i-1])
    ET = np.zeros(P.shape)
    for i in range(1,ns+1):
        ET += (i+1) * (Phi_list[i] - Phi_list[i-1])
    return Phi_list, ET


def vector(index, m):
    
    m = np.zeros(m.size)
    m[index] = 1
    return m


# General function to simulate hitting time for Exercise 2.1
def simulate_hitting_time(P, states, nr):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        states {list[int]} -- the list [start state, end state], index starts from 0
        nr {int} -- largest step to consider

    Returns:
        T {list[int]} -- a size nr list contains the hitting time of all realizations
    '''

    # Add code here to simulate following quantities:
    # T[i] = hitting time of the i-th run (i.e., realization) of process
    # Notice in python the index starts from 0
    start, end = states
    if start == end: return [0] * nr
    T = np.zeros(nr)
    for k in range(nr):
        curr_state = start
        step = 0
        next_state = start
        q = np.zeros(P.shape[0]) 
        while(curr_state != end):
            i = 0
            u = np.random.random_sample()
            curr_vec = vector(curr_state,q)
            cdf_vec = np.cumsum(np.matmul(curr_vec, P))
            if(u < cdf_vec[0]):
                next_state = 0
            while(u >= cdf_vec[i]):
                if(u < cdf_vec[i+1]):
                    next_state = i+1
                i += 1
            step = step + 1
            curr_state = next_state
        T[k] = step
    return T


        
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns



# General function to approximate the stationary distribution of a Markov chain for Exercise 2.4
def stationary_distribution(P):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain

    Returns:
        pi {numpy.array} -- length n, stationary distribution of the Markov chain
    '''

    # Add code here: Think of pi as column vector, solve linear equations:
    #     P^T pi = pi
    #     sum(pi) = 1
    I = np.identity(P.shape[0])
    Transf = (I-P).T
    A = nullspace(Transf)
    A = A/sum(A)
    return A




