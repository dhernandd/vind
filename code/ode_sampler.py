# Copyright 2018 Daniel Hernandez Diaz, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import os
import pickle

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


MODE = 'odegen' # ['odegen', 'inputgen', 'other']
WITH_INPUTS = True
PLOT = True
SAVEROOT = '/Users/danielhernandez/work/supervind/data/'
SAVEDIR = 'pendulumwi002/'
SAVEFILE = 'datadict'
LOAD_INPUTS_DIR = 'inputs1D_001/'
LOAD_INPUTS_FILE = 'inputdict'
ODE = 'pendulum_wi'
NSAMPS = 500
NTBINS = 100
PARAMS = [(0.25, 5.0), (0.5, 3.0)]
Y0_RANGES = np.array([[0, np.pi], [-0.05, 0.05]])
IS_1D = True
MAIN_SCALE = 2.0
DO_SAVE = True

def pendulum(y, _, b, c):
    """
    y = {theta, omega}
    dtheta/dt = omega
    domega/dt = -b*omega - c*sin(theta)
    """
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt
    
def pendulum_winput(y, t, b, c, s, samp, input_key='inputs'):
    """
    y = {theta, omega}
    dtheta/dt = omega
    domega/dt = -b*omega - c*sin(theta) + s*I(t)
    """
    input_file = SAVEROOT + LOAD_INPUTS_DIR + LOAD_INPUTS_FILE
    with open(input_file, 'rb+') as f:
        input_dict = pickle.load(f)
        I = input_dict[input_key][samp]
        T = input_dict['time_points']
        NTbins = len(T)
        range_min, range_max = T[0], T[-1]
        step = (range_max - range_min)/(NTbins-1)

    def get_input(t, I, T):
        """
        Assumes linear intervals in T
        """
        # This function will be called by the ODE integrators with FLOAT t. We
        # first turn this float to an integer index i, that labels the integer
        # sequence of inputs I[i]
        idx = int(t//step)
        
        # The integrator needs to evaluate the ODE in points outside of the range. Just
        # assign to those the last input.
        if idx >= len(T) - 1: idx = len(T) - 2
        
        # For time t, return a linear extrapolation between I[i] and I[i+1]
        return I[idx,0] + (I[idx+1,0]-I[idx,0])*(t-T[idx])/(T[idx+1] - T[idx]) 

    theta, omega = y
#     dydt = [omega - s*get_input(t, I, T), -b*omega - c*np.sin(theta)]
    dydt = [omega - s*np.sin(get_input(t, I, T)), -b*omega - c*np.sin(theta)]
    return dydt


def noisy_integrate(ode, y0, Tbins, ode_params, noise_scale=0.0):
    """
    """
    t = np.linspace(0, 10, Tbins)
    sol = odeint(ode, y0, t, args=ode_params)
    
    noise = np.random.normal(scale=noise_scale, size=sol.shape)
    return sol + noise
    
def noisy_integrate_winputs(ode, y0, ode_params, t, samp, input_scale=1.0,
                            noise_scale=0.0, inputs_key='inputs'):
    """
    """
    ode_params = ode_params + (input_scale, samp, inputs_key)
    sol = odeint(ode, y0, t, args=ode_params)
    
    noise = np.random.normal(scale=noise_scale, size=sol.shape)
    return sol + noise
    
def generate_data(ode, y0ranges, params, Nsamps=200, Tbins=100, noise_scale=0.1,
                  data_scale=1.0, input_scale=1.0):
    """
    """
    num_ids = len(params)

    yDim = len(y0ranges)
    Y0data = []

    for d in range(yDim):
        Y0data.append(np.random.uniform(low=y0ranges[d][0], high=y0ranges[d][1], size=Nsamps))
    Y0 = np.stack(Y0data).T
    
    print('Nsamps', Nsamps)
    if WITH_INPUTS:
        input_file = SAVEROOT + LOAD_INPUTS_DIR + LOAD_INPUTS_FILE
        with open(input_file, 'rb+') as f:
            input_dict = pickle.load(f)
            t = input_dict['time_points']
            I = input_dict['inputs']
            noI = input_dict['noinputs']
            Nsamps_I = len(I)
            Nsamps_noI = len(noI)
            
        # Integrate for no inputs first
        Ydata = np.zeros((Nsamps_noI, Tbins, yDim))
        Ids = np.zeros(Nsamps_noI)
        print('Solving the ODE with no inputs first')
        for samp in range(Nsamps_noI):
            if samp % 10 == 0:
                print('Samples', samp)
            j = np.random.choice(list(range(num_ids)))
            Ydata[samp] = data_scale*noisy_integrate_winputs(ode, Y0[samp], params[j], t, samp,
                                                             input_scale=input_scale,
                                                             noise_scale=noise_scale,
                                                             inputs_key='noinputs')
            Ids[samp] = j
            assert len(Ydata) == len(Ids) == len(I)
        Ytrain, Yvalid, Ytest = ( Ydata[:-Nsamps_noI//5], Ydata[-Nsamps_noI//5:-Nsamps_noI//10],
                                  Ydata[-Nsamps_noI//10:] )
        Idtrain, Idvalid, Idtest = ( Ids[:-Nsamps_noI//5], Ids[-Nsamps_noI//5:-Nsamps_noI//10],
                                     Ids[-Nsamps_noI//10:] )
        noItrain, noIvalid, noItest = ( noI[:-Nsamps_noI//5], noI[-Nsamps_noI//5:-Nsamps_noI//10],
                                  noI[-Nsamps_noI//10:] )
        
        # Integrate with inputs
        Ydata_I = np.zeros((Nsamps_I, Tbins, yDim))
        Ids_I = np.zeros(Nsamps_I)
        print('\nSolving the ODE with inputs')
        for samp in range(Nsamps_I):
            if samp % 10 == 0:
                print('Samples', samp)
            j = np.random.choice(list(range(num_ids)))
            Ydata_I[samp] = data_scale*noisy_integrate_winputs(ode, Y0[samp], params[j], t, samp,
                                                            input_scale=input_scale,
                                                            noise_scale=noise_scale,
                                                            inputs_key='inputs')
            Ids_I[samp] = j
            assert len(Ydata_I) == len(Ids_I) == len(I)
        l_inds = np.arange(len(Ydata))
        np.random.shuffle(l_inds)
        Ytrain_wI, Yvalid_wI, Ytest_wI = ( Ydata_I[l_inds[:-Nsamps_I//5]],
                                           Ydata_I[l_inds[-Nsamps_I//5:-Nsamps_I//10]],
                                           Ydata_I[l_inds[-Nsamps_I//10:]] )
        Idtrain_wI, Idvalid_wI, Idtest_wI = ( Ids_I[l_inds[:-Nsamps_I//5]],
                                              Ids_I[l_inds[-Nsamps_I//5:-Nsamps_I//10]],
                                              Ids_I[l_inds[-Nsamps_I//10:]] )
        Itrain, Ivalid, Itest = ( I[l_inds[:-Nsamps_I//5]],
                                  I[l_inds[-Nsamps_I//5:-Nsamps_I//10]],
                                  I[l_inds[-Nsamps_I//10:]] )

        datadict = {'Ytrain' : Ytrain, 'Yvalid' : Yvalid, 'Ytest' : Ytest,
                    'Idtrain' : Idtrain, 'Idvalid' : Idvalid, 'Idtest' : Idtest,
                    'noItrain' : noItrain, 'noIvalid' : noIvalid, 'noItest' : noItest,
                    'Ytrain_wI' : Ytrain_wI, 'Yvalid_wI' : Yvalid_wI, 'Ytest_wI' : Ytest_wI,
                    'Idtrain_wI' : Idtrain_wI, 'Idvalid_wI' : Idvalid_wI, 'Idtest_wI' : Idtest_wI,
                    'Itrain' : Itrain, 'Ivalid' : Ivalid, 'Itest' : Itest}
    else:
        I = np.zeros((Nsamps, Tbins, 1))
        for samp in range(Nsamps):
            j = np.random.choice(list(range(num_ids)))
            Ydata[samp] = data_scale*noisy_integrate(ode, Y0[samp], Tbins, params[j],
                                                    noise_scale=noise_scale)
            Ids[samp] = j
        Ytrain, Yvalid, Ytest = ( Ydata[:-Nsamps_noI//5], Ydata[-Nsamps_noI//5:-Nsamps_noI//10],
                                  Ydata[-Nsamps_noI//10:] )
        Idtrain, Idvalid, Idtest = ( Ids[:-Nsamps_noI//5], Ids[-Nsamps_noI//5:-Nsamps_noI//10],
                                     Ids[-Nsamps_noI//10:] )
        datadict = {'Ytrain' : Ytrain, 'Yvalid' : Yvalid, 'Ytest' : Ytest,
                    'Idtrain' : Idtrain, 'Idvalid' : Idvalid, 'Idtest' : Idtest}

    return datadict

def generate_input_samps(Nsamps=500, NTbins=100, max_freq=6, main_scale=2.0, new_data_idx=30,
                         reserve_for_zero_input=250, freq_scale=0.5, plot=False):
    """
    """
    Nsamps_winputs = Nsamps - reserve_for_zero_input

    I = np.zeros((Nsamps_winputs, NTbins, 1))
    num_samps_per_freq = (Nsamps_winputs)//max_freq
    T = np.linspace(0, 10, NTbins)
    for samp in range(Nsamps_winputs):
        # Initialize a new sample with a section coming from a lower frequency sample
        starting_t = np.random.randint(1, new_data_idx)
        add_from_samp = np.random.randint(samp+1)
        if samp > num_samps_per_freq and np.random.rand() > 0.5:
            I[samp,:starting_t] = I[add_from_samp,:starting_t]
            I[samp,-starting_t:] = I[add_from_samp,-starting_t:]
        else:
            I[samp,:starting_t] = np.zeros((starting_t, 1))
            I[samp,-starting_t:] = np.zeros((starting_t, 1))
        
        base_freq = samp//num_samps_per_freq
        if base_freq >= 0:
#             t = np.linspace(starting_t, NTbins-starting_t, NTbins-2*starting_t)
            t = T[starting_t:-starting_t]
            phase = 2*np.pi*np.random.random() - np.pi
            this_input = main_scale*np.cos(freq_scale*base_freq*t + phase)
            I[samp,starting_t:-starting_t] = np.expand_dims(this_input, axis=1)
    
    
    skip = 50
    if plot:
        for i in range(0, Nsamps_winputs, skip):
            plt.plot(I[i,:,0])
        plt.show()
    
    inputdict = {'noinputs' : np.zeros((reserve_for_zero_input, NTbins, 1)),
                 'inputs' : I,
                 'time_points' : T}
    
    return inputdict
    
if __name__ == '__main__':
    """
    """
    if MODE == 'odegen':
        odedict = {'pendulum' : pendulum, 'pendulum_wi' : pendulum_winput}
        ddict = generate_data(odedict[ODE], Y0_RANGES, params=PARAMS, Nsamps=NSAMPS, Tbins=NTBINS,
                              noise_scale=0.05)
        data_path = SAVEROOT + SAVEDIR
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        with open(data_path + SAVEFILE, 'wb+') as data_file:
            pickle.dump(ddict, data_file)
    
        data = ddict['Ytrain']
        Idtrain = ddict['Idtrain']
        
        if PLOT:
            for i in range(10):
                c = 'b' if Idtrain[i] == 0 else 'r'
                plt.plot(data[i,:,0], color=c)
    #             plt.plot(data[i,:,1], color='r')
            plt.show()
    elif MODE == 'inputgen':
        inputdict = generate_input_samps(Nsamps=NSAMPS, plot=True, main_scale=MAIN_SCALE)
        T = inputdict['time_points']
        I = inputdict['inputs']
        noI = inputdict['noinputs']
        t_to_idx = {t : i for i, t in enumerate(T)}
        inputdict['t_to_idx'] = t_to_idx
        if DO_SAVE:
            data_path = SAVEROOT + SAVEDIR
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            with open(data_path + SAVEFILE, 'wb+') as data_file:
                pickle.dump(inputdict, data_file)
    elif MODE == 'other':
        loadfile = SAVEROOT + SAVEDIR + SAVEFILE
        
        with open(loadfile, 'rb') as f:
            datadict = pickle.load(f)
        Y = datadict['Ytrain']
        YI = datadict['Ytrain_wI']
        print(Y.shape)
        ymax, nmax = 0.0, 0
        for n, y in enumerate(Y):
            if np.max(Y[n]) > ymax:
                ymax, nmax = np.max(Y[n]), n
        print(np.max(Y), nmax)
        noI = datadict['noItrain']
        I = datadict['Itrain']

        samp = 10
        plt.plot(Y[samp,:,0])
        plt.plot(noI[samp])
        plt.show()
        plt.plot(YI[samp,:,0])
        plt.plot(I[samp])
        plt.show()
        
    
    
    
    