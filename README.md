# VIND: Variational Inference for Nonlinear Dynamics (with tensorflow)

This code is a basic implementation in tensorflow, of the paper "[Variational Inference for Nonlinear Dynamics](https://github.com/dhernandd/vind/blob/master/paper/nips_workshop.pdf)", accepted for the Time Series Workshop at NIPS 2017. It represents a sequential variational autoencoder that is able to infer nonlinear dynamics in the latent space. The training algorithm makes use of a novel, two-step technique for optimization based on the [Fixed Point Iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration) method for finding fixed points of iterative equations.

| Original | Inferred |
|-----------|----------|
|<img src="https://github.com/dhernandd/vind/blob/master/data/gaussian/quiver_plot.png" width="100%" /> | <img src="https://github.com/dhernandd/vind/blob/master/data/gaussian/qplot260.png" width="100%" /> |


# Installation

The code is written in Python 3.5. You will need the bleeding edge versions of the following packages:

- tensorflow
- seaborn

In addition, up-to-date versions of numpy, scipy and matplotlib are expected.

# Usage

Firing `python runner.py` works right off the bat. The code will find a two dimensional encoding and dynamical system describing the provided Gaussian data. A figure is provided with the original dynamical system and simulated trajectories that can be compared with the resulting fit. The hyperparameter `plot2D`, default-set to `True`, will produce these path+dynamics plots automatically for 2D latent spaces.

# TODO

* Work on this README
* Make this implementation more lightweight, currently it still includes a bunch of developer features unnecessary for a first take.
* Make a demonstration notebook
