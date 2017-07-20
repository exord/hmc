import numpy as np
import pylab as plt

import hmc.hmcsampler as hh
import gradients

def bimodal_example(plot=False, seed=1062005):
    niterations = 2000
    np.random.seed(seed)

    # Drawing from mixture
    mu1 = [0,0]
    cov1 = np.eye(2)
    mu2 = [8,0]
    cov2 = np.array([[2,0.8],[0.8,2]])

    def gradu(q):
        return -gradients.gradient_logmixturenormal(mu1, cov1, mu2, cov2,
                                                    0.5, q)

    def u(q):
        return -np.log(gradients.multinormal_pdf(mu1, cov1, q) +
                       gradients.multinormal_pdf(mu2, cov2, q))


    chain, (q,p,h) = hh.sample(niterations, [0,0], u, gradu, 10, 0.3)

    if plot:
        f1 = plt.figure()
        ax = f1.add_subplot(111)
        ax.set_aspect('equal')
        ax.plot(chain[:, 0], chain[:, 1], '.k')
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')

        f2 = plt.figure()
        ax2 = f2.add_subplot(111)
        ax2.set_aspect('equal')
        ax2.set_xlabel(r'$\theta_1$')
        ax2.set_ylabel(r'$\theta_2$')

        f3 = plt.figure()
        ax3 = f3.add_subplot(111)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('H/<H>')

        for i in np.random.randint(niterations, size=20):
            ax2.plot(q[i, :, 0], q[i, :, 1], '.-')
            ax3.plot(h[i]/h[i].mean())

        f4 = plt.figure()
        ax4 = f4.add_subplot(111)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel(r'$\theta_1$ / $\theta_2$')
    
        ax4.plot(chain[:, 1], '-', color='0.5', label=r'$\theta_2$')
        ax4.plot(chain[:, 0], '-k', label=r'$\theta_1$')
        ax4.legend(loc=0)
    return



