from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle

def Lorenz(state, t):
    x = state[0]
    y = state[1]
    z = state[2]

    sigma = 10.0
    rho   = 28.0
    beta  = 8.0/3.0

    xd = sigma * (y-x)
    yd = (rho-z)*x - y
    zd = x*y - beta*z

    # return the state derivs
    return [xd, yd, zd]


def GenerateLorenz(npaths, Dy):

    noise_param = .5
    t = np.arange(0.0, 2.5, 0.01)
    X = np.zeros((npaths,t.shape[0],3))
    GaussianY = np.zeros((npaths, t.shape[0], Dy))
    PoissonY = np.zeros((npaths, t.shape[0], 3))
    fixed_noise = np.random.randn(3, Dy)

    for i in range(npaths):
        init_state = np.random.randint(low=-10, high=10, size=3)
        state = odeint(Lorenz, init_state, t)
        if True:
            # Additive Noise
            noise = noise_param * fixed_noise #np.random.randn(250, 3)
            noisy_state = state + np.random.randn(t.shape[0], 3)
            #GaussianY[i] = state
            GaussianY[i] = np.dot(state, noise) + 0.25*np.random.randn(t.shape[0], Dy)
        if False:
            # Multiplicitive Noise
            noise = np.random.randn(3, Dy)
            GaussianY[i] = np.dot(state, noise)
        rate = np.exp(state/10.)
        PoissonY[i] = np.random.poisson(lam=rate, size=state.shape)


    return GaussianY, PoissonY, state


npaths = 100
GaussY, PoissY, state = GenerateLorenz(npaths, Dy=10)

# Plot Gaussian Data
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(npaths):
    ax.plot(GaussY[i,:,0], GaussY[i,:,1], GaussY[i,:,2])
    ax.scatter(GaussY[i,0,0], GaussY[i,0,1], GaussY[i,0,2])
plt.show()


# Plot Poisson Data
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(npaths):
    ax.plot(PoissY[i,:,0], PoissY[i,:,1], PoissY[i,:,2])
    ax.scatter(PoissY[i,0,0], PoissY[i,0,1], PoissY[i,0,2])
plt.show()

datadict = {'Ytrain': GaussY[0:80],
            'Yvalid': GaussY[80:100],
            'Xtrain': state[0:80],
            'Xvalid': state[80:100]
            }

print(datadict['Ytrain'].shape,
      datadict['Yvalid'].shape)

datadict2 = {'Ytrain': PoissY[0:80],
            'Yvalid': PoissY[80:100],
            'Xtrain': state[0:80],
            'Xvalid': state[80:100]
            }

print(datadict2['Ytrain'].shape,
      datadict2['Yvalid'].shape)

with open('lorenzdataP', 'wb') as handle:
    pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('lorenzdataG', 'wb') as handle:
    pickle.dump(datadict2, handle, protocol=pickle.HIGHEST_PROTOCOL)