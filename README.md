# VIND: Variational Inference for Nonlinear Dynamics (with tensorflow)


This code is the tensorflow, augmented implementation for the paper "[Variational Inference for Nonlinear Dynamics](https://github.com/dhernandd/vind/blob/master/paper/nips_workshop.pdf)", accepted for the Time Series Workshop at NIPS 2017. It represents a sequential variational autoencoder that is able to infer nonlinear dynamics in the latent space. The training algorithm makes use of a novel, two-step technique for optimization based on the [Fixed Point Iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration) method for finding fixed points of iterative equations.

# Installation

The code is written in Python 3.5. You will need the bleeding edge versions of the following packages:

- tensorflow
- seaborn

In addition, up-to-date versions of numpy, scipy and matplotlib are expected.

# Running the code

With the above requirements in place, firing `python runner.py` should work right off the bat. The code will begin finding a two dimensional dynamical system to encode the provided 10D Poisson data. For your convenience, synthetic Gaussian data is also given. 


