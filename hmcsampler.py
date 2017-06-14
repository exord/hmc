from cobmcmc import Sampler
import numpy as np

class HamiltoneanSampler(Sampler):
    """
    A class implementing Hamiltonean Monte Carlo
    """

    def __init__(self, dim, u, grad_u, k):
        """
        """
        super(Sampler, self).__init__(alpha)


def sample(niterations, q0, u, grad_u, nsteps, epsilon):
    """
    """
    dim = len(q0)
    chain = np.empty((niterations, dim))
    chain[0] = q0

    q = np.zeros((niterations, nsteps, dim))
    p = np.zeros((niterations, nsteps, dim))
    h = np.zeros((niterations, nsteps))

    for i in range(1, niterations):

        # Randomly draw momentum variables from canonical distribution
        pi = np.random.randn(dim)
        qi = chain[i - 1]

        # Set first step of orbit
        p[i, 0] = pi
        q[i, 0] = qi

        h[i, 0] = 0.5 * np.sum(pi**2) + u(qi)

        for j in range(1, nsteps):

            p2 = p[i, j-1] - 0.5 * epsilon * grad_u(q[i, j-1])
            q[i, j] = q[i, j-1] + epsilon * p2
            p[i, j] = p2 - 0.5 * epsilon * grad_u(q[i, j])
            h[i, j] = 0.5 * np.sum(p[i, j]**2) + u(q[i, j])

        # Metropolis ratio
        r = np.exp(h[i, 0] - h[i, nsteps-1])

        # Draw random
        if np.random.rand() < r:
            chain[i] = q[i, nsteps-1]
        else:
            chain[i] = chain[i-1]

    return chain, (q, p, h)

        

            
        
                     
    
