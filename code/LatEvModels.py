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
import random

import numpy as np

import tensorflow as tf

# Hideous hack to have this code run both as a package and imported from a
# Jupyter notebook. A fairy dies in Neverland every time you run this.
if __name__ == 'LatEvModels':
    from datetools import addDateTime #@UnresolvedImport #@UnusedImport
    from layers import FullLayer #@UnresolvedImport #@UnusedImport
else:
    from .datetools import addDateTime # @UnresolvedImport @Reimport
    from .layers import FullLayer  # @Reimport


TEST_DIR = './tests/test_results/'
DTYPE = tf.float32

def flow_modulator(x, x0=30.0, a=0.08):
    return (1 - np.tanh(a*(x - x0)))/2

def flow_modulator_tf(X, x0=30.0, a=0.08):
    return tf.cast((1.0 - tf.tanh(a*(X - x0)) )/2, DTYPE)


class NoisyEvolution():
    """
    The Locally Linear Evolution Model. Several possibilities depending on a
    bunch of boolean hyperparams.
    
    i) with_inputs = False:
        x_{t+1} = A(x_t)x_t + eps
    ii) with_inputs = True:
        a) with_additive_shock = True
            x_{t+1} = A(x_t)x_t + f(I_t) + eps
        b) with_mod_dynamics = True
            x_{t+1} = A(x_t, I_t)x_t + eps
    
    eps is Gaussian noise.
    
    The following methods should be implemented
        sample_X:    Draws samples from the Markov Chain.
        compute_LogDensity_Xterms: Computes the loglikelihood of the chain.
    """
    def __init__(self, X, params, Ids=None, Inputs=None):
        """
        Args:
            X : tf.placeholder for the latent sequence
            params: The hyperparameters
            Ids: Possible known identities of the trials (e.g., cell types)
            Inputs: Possible known inputs at each time point.
        """        
        self.X = X
        self.params = params
        
        # The Ids placeholder
        self.Ids = Ids = ( tf.placeholder(dtype=tf.int32, shape=[None], name='Ids') 
                           if Ids is None else Ids )
        
        self.xDim = xDim = params.xDim
        self.pDim = pDim = params.pDim
        self.Nsamps = tf.shape(self.X)[0]
        self.NTbins = tf.shape(self.X)[1]
        if hasattr(params, 'num_diff_entities'):
            self.num_diff_entities = params.num_diff_entities
        else: self.num_diff_entities = 1

        # The pDim parameters corresponding to each different entity.
        self.ev_params_Pxp = tf.get_variable('ev_params', shape=[self.num_diff_entities, pDim])

        if params.with_inputs:
            # The Inputs placeholder (if with_inputs)
            self.iDim = iDim = params.iDim
            self.I = ( tf.placeholder(dtype=DTYPE, shape=[None, None, iDim], name='Inputs') if 
                       Inputs is None else Inputs )
        
        # Variance (Q) of the state-space evolution. 
        init_Q = params.init_Q
        self.QInvChol_dxd = tf.get_variable('QInvChol', 
                                            initializer=tf.cast(init_Q*tf.eye(xDim), DTYPE),
                                            trainable=params.is_Q_trainable)
        self.QChol_dxd = tf.matrix_inverse(self.QInvChol_dxd, name='QChol')
        self.QInv_dxd = tf.matmul(self.QInvChol_dxd, self.QInvChol_dxd, transpose_b=True,
                                  name='QInv')
        self.Q_dxd = tf.matmul(self.QChol_dxd, self.QChol_dxd, transpose_b=True,
                               name='Q')
        
        # Variance of the initial points
        init_Q0 = params.init_Q0
        self.Q0InvChol_dxd = tf.get_variable('Q0InvChol',
                                             initializer=tf.cast(init_Q0*tf.eye(xDim), DTYPE))
        self.Q0Chol_dxd = tf.matrix_inverse(self.Q0InvChol_dxd, name='Q0Chol')
        self.Q0Inv_dxd = tf.matmul(self.Q0InvChol_dxd, self.Q0InvChol_dxd,
                                   transpose_b=True, name='Q0Inv')
        
        # The mean starting coordinates in state-space
        self.x0 = tf.get_variable('x0', initializer=tf.cast(tf.zeros(self.xDim), DTYPE))
         
        self.alpha = params.alpha

        # The base linear element of the evolution 
        if not hasattr(self, 'Alinear'):
            self.Alinear_dxd = tf.get_variable('Alinear', initializer=tf.eye(xDim),
                                               dtype=DTYPE)
        
        # Define the evolution for *this* instance
#         self.A_NxTxdxd, self.Awinflow_NxTxdxd, self.B_NxTxdxd = self._define_evolution_network()
        self.A_NxTxdxd, self.Awinflow_NxTxdxd, self.B_NxTxdxd = self._define_evolution_network_wi()
    
    def _define_evolution_network(self, X=None, Ids=None):
        """
        
        """
        xDim = self.xDim
        pDim = self.pDim
        rDim = xDim + pDim
        
        if X is None and Ids is not None: raise ValueError("Must provide an X for these Ids")
        X_NxTxd = self.X if X is None else X
        if Ids is None: Ids = self.Ids            

        Nsamps = tf.shape(X_NxTxd)[0]
        NTbins = tf.shape(X_NxTxd)[1]
        
        # Expand the parameters according to the provided trial Ids
        ev_params_Nxp = tf.gather(self.ev_params_Pxp, indices=Ids)
        ev_params_NxTxp = tf.tile(tf.expand_dims(ev_params_Nxp, axis=1), [1, NTbins, 1])

        rangeB = self.params.initrange_B
        evnodes = 200
        # Concatenate the parameters to the state at time t
        State_NxTxr = tf.concat([X_NxTxd, ev_params_NxTxp], axis=2)
        State_NTxr = tf.reshape(State_NxTxr, [Nsamps*NTbins, rDim])
        fully_connected_layer = FullLayer(collections=['EVOLUTION_PARS'])
        with tf.variable_scope("ev_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(State_NTxr, evnodes, 'softmax', 'full1')
            full2 = fully_connected_layer(full1, evnodes//2, 'softplus', 'full2',
                                          initializer=tf.orthogonal_initializer())
            output = fully_connected_layer(full2, xDim**2, nl='linear', scope='output',
                                           initializer=tf.random_uniform_initializer(-rangeB, rangeB))
        B_NxTxdxd = tf.reshape(output, [Nsamps, NTbins, xDim, xDim], name='B')
        B_NTxdxd = tf.reshape(output, [Nsamps*NTbins, xDim, xDim])
        
        A_NTxdxd = self.alpha*B_NTxdxd + self.Alinear_dxd # Broadcast
        A_NxTxdxd = tf.reshape(A_NTxdxd, [Nsamps, NTbins, xDim, xDim], name='A')
        
        X_norms = tf.norm(State_NTxr[:,:xDim], axis=1)
        fl_mod = flow_modulator_tf(X_norms)
        eye_swap = tf.transpose(tf.tile(tf.expand_dims(tf.eye(self.xDim), 0),
                                        [Nsamps*NTbins, 1, 1]), [2,1,0])
        Awinflow_NTxdxd = tf.transpose(fl_mod*tf.transpose(
                                A_NTxdxd, [2,1,0]) + 0.9*(1.0-fl_mod)*eye_swap, [2,1,0])
        Awinflow_NxTxdxd = tf.reshape(Awinflow_NTxdxd, 
                                      [Nsamps, NTbins, xDim, xDim], name='Awinflow')
        
        
        return A_NxTxdxd, Awinflow_NxTxdxd, B_NxTxdxd

    def _define_evolution_network_wi(self, X=None, Ids=None, Inputs=None):
        """
        
        """
        params = self.params
        
        xDim = self.xDim
        pDim = self.pDim
        if params.with_inputs:
            iDim = self.iDim    
        rDim = xDim + pDim + iDim if params.with_mod_dynamics else xDim + pDim
        
        if X is None and Ids is not None: raise ValueError("Must provide an X for these Ids")
        X_NxTxd = self.X if X is None else X
        if Ids is None: Ids = self.Ids

        Nsamps = tf.shape(X_NxTxd)[0]
        NTbins = tf.shape(X_NxTxd)[1]
        # Expand the parameters according to the provided trial Ids
        ev_params_Nxp = tf.gather(self.ev_params_Pxp, indices=Ids)
        ev_params_NxTxp = tf.tile(tf.expand_dims(ev_params_Nxp, axis=1), [1, NTbins, 1])

        if params.with_mod_dynamics:
            Inputs_NxTxi = self.I if Inputs is None else Inputs

        rangeB = self.params.initrange_B
        evnodes = 200
        # Concatenate the parameters to the state at time t
        
        State_NxTxr = ( tf.concat([X_NxTxd, ev_params_NxTxp, Inputs_NxTxi], axis=2)
                        if params.with_mod_dynamics else
                        tf.concat([X_NxTxd, ev_params_NxTxp], axis=2) )
        State_NTxr = tf.reshape(State_NxTxr, [Nsamps*NTbins, rDim])
        fully_connected_layer = FullLayer(collections=['EVOLUTION_PARS'])
        with tf.variable_scope("ev_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(State_NTxr, evnodes, 'softmax', 'full1')
            full2 = fully_connected_layer(full1, evnodes//2, 'softplus', 'full2',
                                          initializer=tf.orthogonal_initializer())
            output = fully_connected_layer(full2, xDim**2, nl='linear', scope='output',
                                           initializer=tf.random_uniform_initializer(-rangeB, rangeB))
        B_NxTxdxd = tf.reshape(output, [Nsamps, NTbins, xDim, xDim], name='B')
        B_NTxdxd = tf.reshape(output, [Nsamps*NTbins, xDim, xDim])
        
        A_NTxdxd = self.alpha*B_NTxdxd + self.Alinear_dxd # Broadcast
        A_NxTxdxd = tf.reshape(A_NTxdxd, [Nsamps, NTbins, xDim, xDim], name='A')
        
        X_norms = tf.norm(State_NTxr[:,:xDim], axis=1)
        fl_mod = flow_modulator_tf(X_norms)
        eye_swap = tf.transpose(tf.tile(tf.expand_dims(tf.eye(self.xDim), 0),
                                        [Nsamps*NTbins, 1, 1]), [2,1,0])
        Awinflow_NTxdxd = tf.transpose(fl_mod*tf.transpose(
                                A_NTxdxd, [2,1,0]) + 0.9*(1.0-fl_mod)*eye_swap, [2,1,0])
        Awinflow_NxTxdxd = tf.reshape(Awinflow_NTxdxd, 
                                      [Nsamps, NTbins, xDim, xDim], name='Awinflow')
         
        return A_NxTxdxd, Awinflow_NxTxdxd, B_NxTxdxd

    def get_A_grads(self, xin=None, idin=None, iin=None):
        xDim = self.xDim
        if xin is None: xin = self.x
        
        singleA_1x1xdxd = ( self._define_evolution_network_wi(xin, idin, iin)[0]
                            if self.params.with_mod_dynamics else
                            self._define_evolution_network_wi(xin, idin)[0] )
        singleA_d2 = tf.reshape(singleA_1x1xdxd, [xDim**2])
        grad_list_d2xd = tf.squeeze(tf.stack([tf.gradients(Ai, xin) for Ai
                                              in tf.unstack(singleA_d2)]))

        return grad_list_d2xd 

    def eval_nextX(self, session, Xdata, Xvar_name='X:0', scope="", with_inflow=False, Id=0):
        """
        Given a symbolic array of points in latent space Xdata = [X0, X1,...,XT], \
        gives the prediction for the next time point
         
        This is useful for the quiver plots.
        
        Args:
            session : The external tf.Session
            Xdata : Points in latent space at which the dynamics A(X) shall be
            determined.
            X_var_name : The name of the tensorflow node
            with_inflow : Should an inward flow from infinity be superimposed to A(X)?
        """
        Nsamps, Tbins = Xdata.shape[0], Xdata.shape[1]
        
        Iddata = np.full(len(Xdata), Id, dtype=np.int32)
        totalA = self.A_NxTxdxd if not with_inflow else self.Awinflow_NxTxdxd
        
        if self.params.with_mod_dynamics:
            Inputdata = np.zeros([Nsamps, Tbins, self.iDim])
            A = session.run(totalA, feed_dict={scope+Xvar_name : Xdata,
                                               scope+'Ids:0' : Iddata,
                                               scope+'Inputs:0' : Inputdata})
        else:
            A = session.run(totalA, feed_dict={scope+Xvar_name : Xdata,
                                               scope+'Ids:0' : Iddata})
        A = A[:,:-1,:,:].reshape(Nsamps*(Tbins-1), self.xDim, self.xDim)
        Xdata = Xdata[:,:-1,:].reshape(Nsamps*(Tbins-1), self.xDim)
                
        return np.einsum('ij,ijk->ik', Xdata, A).reshape(Nsamps, Tbins-1, self.xDim)
    
    @staticmethod
    def define2DLattice(x1range=(-30.0, 30.0), x2range=(-30.0, 30.0)):
        x1coords = np.linspace(x1range[0], x1range[1])
        x2coords = np.linspace(x2range[0], x2range[1])
        Xlattice = np.array(np.meshgrid(x1coords, x2coords))
        return Xlattice.reshape(2, -1).T

    def quiver2D_flow(self, session, Xvar_name='X:0', scope="", clr='black', scale=25,
                      x1range=(-35.0, 35.0), x2range=(-35.0, 35.0), figsize=(13,13), 
                      pause=False, draw=False, with_inflow=False, newfig=True, savefile=None,
                      Id=0):
        """
        Draws a quiver plot representing the hidden dynamics when the latent
        space is 2D.
        """
        import matplotlib.pyplot as plt
        if newfig:
            plt.ion()
            plt.figure(figsize=figsize)
        lattice = self.define2DLattice(x1range, x2range)
        Tbins = lattice.shape[0]
        lattice = np.reshape(lattice, [1, Tbins, self.xDim])
        
        nextX = self.eval_nextX(session, lattice, Xvar_name=Xvar_name, scope=scope,
                                with_inflow=with_inflow, Id=Id)
        nextX = nextX.reshape(Tbins-1, self.xDim)
        X = lattice[:,:-1,:].reshape(Tbins-1, self.xDim)

        plt.quiver(X.T[0], X.T[1], nextX.T[0]-X.T[0], nextX.T[1]-X.T[1], 
                   color=clr, scale=scale)
        axes = plt.gca()
        axes.set_xlim(x1range)
        axes.set_ylim(x2range)
        if draw: plt.draw()  
        
        if pause:
            plt.pause(0.001)
            input('Press Enter to continue.')
        
        if savefile is not None:
            plt.savefig(savefile)
    
    def plot2D_sampleX(self, Xdata, figsize=(13, 13), newfig=True, 
                       pause=True, draw=True, skipped=1):
        """
        Plots the evolution of the dynamical system in a 2D projection..
        """
        import matplotlib.pyplot as plt
        
        ctr = 0
        if newfig:
            plt.ion()
            plt.figure(figsize=figsize)
        for samp in Xdata:
            if ctr % skipped == 0:
                plt.plot(samp[:,0], samp[:,1], linewidth=2)
                plt.plot(samp[0,0], samp[0,1], 'bo')
                axes = plt.gca()
            ctr += 1
        if draw: plt.draw()  
        if pause:
            plt.pause(0.001)
            input('Press Enter to continue.')
            
        return axes
    
    def plot_2Dquiver_paths(self, session, Xdata, Xvar_name='X:0', scope="", 
                            rlt_dir=TEST_DIR+addDateTime()+'/', 
                            rslt_file='quiver_plot', with_inflow=False, savefig=False, draw=False,
                            pause=False, skipped=1, feed_range=True, range_xs=20.0, Id=0):
        """
        Plots a superposition of the 2D quiver plot and the paths in latent
        space. Useful to check that the trajectories roughly follow the
        dynamics.
        """
        if savefig:
            if not os.path.exists(rlt_dir): os.makedirs(rlt_dir)
            rslt_file = rlt_dir + rslt_file
        
        import matplotlib.pyplot as plt
        axes = self.plot2D_sampleX(Xdata, pause=pause, draw=draw, newfig=True, skipped=skipped)
        if feed_range:
            x1range, x2range = axes.get_xlim(), axes.get_ylim()
            s = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
        else:
            x1range = x2range = (-range_xs, range_xs)
            s = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
        
        self.quiver2D_flow(session, Xvar_name=Xvar_name, scope=scope, pause=pause, 
                           x1range=x1range, x2range=x2range, scale=s, newfig=False, 
                           with_inflow=with_inflow, draw=draw, Id=Id)
        if savefig:
            plt.savefig(rslt_file)
        else:
            pass
        plt.close()


class LocallyLinearEvolution(NoisyEvolution):
    """
    The Locally Linear Evolution Model:
        x_{t+1} = A(x_t)x_t + eps
    where eps is Gaussian noise.
    
    An evolution model, it should implement the following key methods:
        sample_X:    Draws samples from the Markov Chain.
        compute_LogDensity_Xterms: Computes the loglikelihood of the chain.
    """
    def __init__(self, X, params, Ids=None, Inputs=None):
        """
        """
        NoisyEvolution.__init__(self, X, params, Ids=Ids, Inputs=Inputs)
        
        if params.with_inputs:
            self.Iterm_NxTxd = self._define_input_to_latent()
            
        self.logdensity_Xterms, self.checks_LX = self.compute_LogDensity_Xterms()
    
    def _define_input_to_latent(self, IInput=None):
        """
        Defines the map f(I_t) in the evolution equation
        
        X_{t+1} = A(X_t, I_t)X_t + f(I_t) 
        
        from the input to the state space shock. Note that this type of shock is
        rather limited, in particular the effect of the additive term does not
        depend on the current state. The presence of an input I_t adds the
        same value f(I_t) no matter where the system is.
        """
        iDim = self.iDim
        xDim = self.xDim
        
        IInput_NxTxi = self.I if IInput is None else IInput  
        Nsamps = tf.shape(IInput_NxTxi)[0] 
        NTbins = tf.shape(IInput_NxTxi)[1]
        Ids = self.Ids
        
        self.input_params_p = tf.get_variable('input_params', shape=[self.num_diff_entities],)
        tf.add_to_collection('INPUT', self.input_params_p)
        input_params_N = tf.gather(self.input_params_p, indices=Ids)
        
        IInput_NTxi = tf.reshape(IInput_NxTxi, [Nsamps*NTbins, iDim])
        fully_connected_layer = FullLayer(collections=['INPUT'])
        with tf.variable_scope("input_nn", reuse=tf.AUTO_REUSE):
            full1 = fully_connected_layer(IInput_NTxi, 128, 'relu', 'full1')
            full = fully_connected_layer(full1, xDim, 'linear', 'full',
                                         initializer=tf.random_normal_initializer(stddev=0.1))

        # put sample dimension last to broadcast
        Iterm_dxTxN = tf.transpose(tf.reshape(full, [Nsamps, NTbins, xDim]), [2,1,0])
        Iterm_NxTxd = tf.transpose(input_params_N*Iterm_dxTxN, [2,1,0], name='Iterm') 
        return Iterm_NxTxd

    def compute_LogDensity_Xterms(self, X=None, with_inflow=False):
        """
        Computes the X terms of log p(X, Y) (joint hidden state/observation
        loglikelihood).

        No need to pass Ids, Inputs here, this always uses the properties of
        self.
        
        Args:
        """
        xDim = self.xDim
        X_NxTxd = self.X if X is None else X
        if X is None:
            A_NxTxdxd = self.A_NxTxdxd if not with_inflow else self.Awinflow_NxTxdxd
        else:
#             A_NxTxdxd, Awinflow_NxTxdxd, _ = self._define_evolution_network(X_NxTxd)
            A_NxTxdxd, Awinflow_NxTxdxd, _ = self._define_evolution_network_wi(X_NxTxd)
            A_NxTxdxd = A_NxTxdxd if not with_inflow else Awinflow_NxTxdxd
        Nsamps = tf.shape(X_NxTxd)[0]
        NTbins = tf.shape(X_NxTxd)[1]
        
        Xprime_NxTm1xd = tf.squeeze(tf.matmul(tf.expand_dims(X_NxTxd[:,:-1], axis=2),
                                              A_NxTxdxd[:,:-1]))
        resX_NxTm1x1xd = tf.expand_dims(X_NxTxd[:,1:] - Xprime_NxTm1xd, axis=2)
        resX0_Nxd = X_NxTxd[:,0] - self.x0
        if self.params.with_inputs and self.params.with_Iterm:
            Iterm_NxTx1xd = tf.expand_dims(self.Iterm_NxTxd, axis=2)
            resX_NxTm1x1xd = resX_NxTm1x1xd - Iterm_NxTx1xd[:,:-1]
            
        QInv_NxTm1xdxd = tf.tile(tf.reshape(self.QInv_dxd, [1, 1, xDim, xDim]),
                                 [Nsamps, NTbins-1, 1, 1])

        LX1 = -0.5*tf.reduce_sum(resX0_Nxd*tf.matmul(resX0_Nxd, self.Q0Inv_dxd), name='LX0')
        LX2 = -0.5*tf.reduce_sum(resX_NxTm1x1xd*tf.matmul(resX_NxTm1x1xd, QInv_NxTm1xdxd), name='L2')
        LX3 = 0.5*tf.log(tf.matrix_determinant(self.Q0Inv_dxd))*tf.cast(Nsamps, DTYPE)
        LX4 = 0.5*tf.log(tf.matrix_determinant(self.QInv_dxd))*tf.cast(Nsamps*(NTbins-1), DTYPE)
        LX5 = -0.5*np.log(2*np.pi)*tf.cast(Nsamps*NTbins*xDim, DTYPE)
        
        LatentDensity = LX1 + LX2 + LX3 + LX4 + LX5
        
        return LatentDensity, [LX1, LX2, LX3, LX4, LX5]
    

    #** The methods below take a session as input and are not part of the main
    #** graph. They should only be used standalone.

    def sample_X(self, sess, Xvar_name, Nsamps=2, NTbins=3, X0data=None, with_inflow=False,
                 path_mse_threshold=0.1, draw_plots=False, init_variables=True, num_ids=1):
        """
        Runs forward the stochastic model for the latent space.
         
        Returns a numpy array of samples
        """
        print('Sampling from latent dynamics...')
        if init_variables: 
            sess.run(tf.global_variables_initializer())
        
        xDim = self.xDim
        Q0Chol = sess.run(self.Q0Chol_dxd)
        QChol = sess.run(self.QChol_dxd)
        Nsamps = X0data.shape[0] if X0data is not None else Nsamps
        Xdata_NxTxd = np.zeros([Nsamps, NTbins, self.xDim])
        x0scale = 15.0
        
        A_NxTxdxd = self.Awinflow_NxTxdxd if with_inflow else self.A_NxTxdxd
        A_NTxdxd = tf.reshape(A_NxTxdxd, shape=[-1, xDim, xDim])
        for samp in range(Nsamps):
            samp_norm = 0.0 # needed to avoid paths that start too close to an attractor
            
            # lower path_mse_threshold to keep paths closer to trivial
            # trajectories, x = const.
            this_id = random.randint(1, num_ids)
            while samp_norm < path_mse_threshold:
                X_single_samp_1xTxd = np.zeros([1, NTbins, self.xDim])
                x0 = ( x0scale*np.dot(np.random.randn(self.xDim), Q0Chol) if 
                       X0data is None else X0data[samp] )
                X_single_samp_1xTxd[0,0] = x0
                
                noise_samps = np.random.randn(NTbins, self.xDim)
                for curr_tbin in range(NTbins-1):
                    curr_X_1x1xd = X_single_samp_1xTxd[:,curr_tbin:curr_tbin+1,:]
                    A_1xdxd = sess.run(A_NTxdxd, feed_dict={Xvar_name : curr_X_1x1xd,
                                                            'Ids:0' : np.array([this_id])})
                    A_dxd = np.squeeze(A_1xdxd, axis=0)
                    X_single_samp_1xTxd[0,curr_tbin+1,:] = ( 
                        np.dot(X_single_samp_1xTxd[0,curr_tbin,:], A_dxd) + 
                        np.dot(noise_samps[curr_tbin+1], QChol) )
            
                # Compute MSE and discard path is MSE < path_mse_threshold
                # (trivial paths)
                Xsamp_mse = np.mean([np.linalg.norm(X_single_samp_1xTxd[0,tbin+1] - 
                                    X_single_samp_1xTxd[0,tbin]) for tbin in 
                                                    range(NTbins-1)])
                samp_norm = Xsamp_mse
        
            Xdata_NxTxd[samp,:,:] = X_single_samp_1xTxd
        
        if draw_plots:
            self.plot_2Dquiver_paths(sess, Xdata_NxTxd, Xvar_name, with_inflow=with_inflow)

        return Xdata_NxTxd        
    
        