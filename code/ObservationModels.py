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
from abc import abstractmethod

import numpy as np

import tensorflow as tf

# pylint: disable=bad-indentation, no-member, protected-access

# Hideous hack to have this code run both as a package and imported from a
# Jupyter notebook. The lines below were first seen scratched in the walls of
# Alcatraz.
if __name__ == 'ObservationModels':
    from layers import FullLayer  # @UnresolvedImport @UnusedImport pylint: disable=import-error
else:
    from .layers import FullLayer  # @Reimport

TEST_DIR = './tests/test_results/'

DTYPE = tf.float32

class ObsModel():
    """
    Abstract class for the observation models.
    
    An ObsModel must define the following methods:
    
    compute_LogDensity: Computes symbolically the autoencoder part of the
        LogDensity (in general of the form d(Y, Y') where d represents some
        measure of the distance between Y and Y') and calls the method from
        self.lat_ev_model that computes the contribution from the X-terms.
    
    sample_XY: Generates a (Y, X) sample
    """
    def __init__(self, Y, X, params, lat_ev_model):
        """
        Initialize the ObsModel
        
        Args:
            X (tf.Placeholder): The latent state tensor.
            Y (tf.Placeholder): The observations tensor. 
            params (tf..flags): Tensorflow flags containing the
                hyperparameters of the model
            lat_ev_model (NoisyEvolution): The latent evolution model.
        """
        self.X = X
        self.Y = Y
        self.yDim = params.yDim
        self.xDim = params.xDim
        self.Nsamps = tf.shape(self.X)[0]
        self.NTbins = tf.shape(self.X)[1]

        self.params = params

        self.lat_ev_model = lat_ev_model
        
    @abstractmethod
    def compute_LogDensity(self, InputX=None, with_inflow=False):
        """
        Compute the LogDensity 
        """
        raise NotImplementedError("This is an abstract method. Please define it in "
                                  "the children classes")
    
    @abstractmethod
    def sample_XY(self):
        """
        Sample (Y, X)
        """
        raise NotImplementedError("This is an abstract method. Please define it in "
                                  "the children classes")


class PoissonObs(ObsModel):
    """
    A Poisson observation model
    """
    def __init__(self, Y, X, params, lat_ev_model):
        """
        Initialize a Poisson Observation Model
        """
        super(PoissonObs, self).__init__(Y, X, params, lat_ev_model)

        self.rate_NTxD = self._define_rate()
        
        # self.checks is useful for debugging
        self.LogDensity, self.checks = self.compute_LogDensity() 
    
    def _define_rate(self, Input=None):
        """
        Define the map lambda(x), the rate of the Poisson observations as
        a function of the latent state. 
        
        If params.poisson_is_out_positive is positive, the output of the map is
        positive definite through a softplus nonlinearity. Otherwise, an
        exponential activation is applied.
        """
        params = self.params
        if Input is None: Input = self.X
        
        Nsamps = tf.shape(Input)[0]
        NTbins = tf.shape(Input)[1]
        xDim = self.xDim
        yDim = self.yDim
        Input = tf.reshape(Input, [Nsamps*NTbins, xDim], name='X_input')
        
        rangeY = params.initrange_outY
        self.inv_tau = inv_tau = params.poisson_inv_tau
        obs_nodes = 64
        fully_connected_layer = FullLayer()
        with tf.variable_scope("obs_nn", reuse=tf.AUTO_REUSE):
            if params.is_linear_output:
                full = fully_connected_layer(Input, yDim, 'linear', scope='output')
                rate_NTxD = tf.exp(inv_tau*full)
            else:
                full1 = fully_connected_layer(Input, obs_nodes, 'softplus', 'full1')
                full2 = fully_connected_layer(full1, obs_nodes, 'softplus', 'full2')
                if params.poisson_is_out_positive:
                    rate_NTxD = fully_connected_layer(full2, yDim, 'softplus', scope='output',
                                  b_initializer=tf.random_normal_initializer(1.0, rangeY))
                else:
                    full3 = fully_connected_layer(full2, yDim, 'linear', scope='output')
    #                            initializer=tf.random_uniform_initializer(-rangeY, rangeY))
                    rate_NTxD = tf.exp(inv_tau*full3)
            self.rate_NxTxD = tf.reshape(rate_NTxD, [Nsamps, NTbins, yDim], name='outY') 
            
        return rate_NTxD
        
    def compute_LogDensity(self, InputX=None, with_inflow=False):
        """
        Compute the total LogDensity for a Generative Model with Poisson Observations
        
        Args:
            InputX (tf.Tensor):
            with_inflow (bool):
            
        Returns:
            LD: The total Logdensity, composed of the sum of the X-terms and the
                Y-terms
            checks: Individual terms forming the latent density. This is useful
                to micromanage the training algorithm, catch difficult bugs, and
                in general, better understand what's going on.
        """
        yDim = self.yDim
        if InputX is None:
            Nsamps = self.Nsamps
            NTbins = self.NTbins
            X = self.X
            LX, Xchecks = self.lat_ev_model.compute_LogDensity_Xterms(with_inflow=with_inflow)
            rate_NTxD = self.rate_NTxD
        else:
            Nsamps = tf.shape(InputX)[0]
            NTbins = tf.shape(InputX)[1]
            X = InputX
            LX, Xchecks = self.lat_ev_model.compute_LogDensity_Xterms(X, 
                                                                with_inflow=with_inflow)        
            rate_NTxD = tf.identity(self._define_rate(X), name='rate_'+X.name[:-2])
        
        Y_NTxD = tf.reshape(self.Y, [Nsamps*NTbins, yDim])
        LY1 = tf.reduce_sum(Y_NTxD*tf.log(rate_NTxD))
        LY2 = tf.reduce_sum(-rate_NTxD)  #pylint: disable=invalid-unary-operand-type
        LY3 = tf.reduce_sum(-tf.lgamma(Y_NTxD + 1.0))
        LY = LY1 + LY2 + LY3
        
        tf.summary.scalar('LogDensity_Yterms', LY) 
        self.LY1_summ = tf.summary.scalar('LY1', LY1)
        
        checks = [LY, LX, LY1, LY2, LY3]
        checks.extend(Xchecks)
        
        LD = tf.add(LX, LY, name='LogDensity')
        return LD, checks 


    #** These methods take a session as input and are not part of the main
    #** graph. They are meant to be used as standalone.
    
    def sample_XY(self, sess, Xvar_name='VAEC/X:0', Nsamps=50,
                  NTbins=100, X0data=None, with_inflow=True, 
                  path_mse_threshold=1.0, draw_plots=False, 
                  init_variables=False):
        """
        Sample from the generative model (Y, X)
        
        Args:
        
        Returns:
            Ydata_NxTxD: Sample observations 
            Xdata_NxTxd: Sample latent states
        """
        if init_variables:
            sess.run(tf.global_variables_initializer())
            
        Xdata_NxTxd = self.lat_ev_model.sample_X(sess, Xvar_name, 
                                                 Nsamps=Nsamps,
                                                 NTbins=NTbins,
                                                 X0data=X0data,
                                                 with_inflow=with_inflow, 
                                                 path_mse_threshold=path_mse_threshold, 
                                                 draw_plots=draw_plots,
                                                 init_variables=init_variables)
        
        rate_NTxD = self.rate_NTxD
        rate = sess.run(rate_NTxD, feed_dict={Xvar_name : Xdata_NxTxd})
        rate = np.reshape(rate, [Nsamps, NTbins, self.yDim])
        Ydata_NxTxD = np.random.poisson(rate)
        
        return Ydata_NxTxD, Xdata_NxTxd
    
    
class GaussianObs(ObsModel):
    """
    A Gaussian observation model
    """
    def __init__(self, Y, X, params, lat_ev_model):
        """
        Initialize a Gaussian Observation Model
        """
        super(GaussianObs, self).__init__(Y, X, params, lat_ev_model)
        
        self.MuY_NxTxD, self.SigmaInvY_DxD = self._define_mean_variance()
        self.LogDensity, self.checks = self.compute_LogDensity() # self.checks meant for debugging
    
    def _define_mean_variance(self, Input=None):
        """
        Define the mean and variance of the Gaussian Observation Model
        
        Returns:
            A pair containing:
            
            - MuY_NxTxD: The means of the Gaussian Observation Model
            - SigmaInv_DxD: The variance of the Gaussian Observation Model
        """
        params = self.params
        if Input is None: Input = self.X
        
        xDim = self.xDim
        yDim = self.yDim
        Nsamps = tf.shape(Input)[0]
        NTbins = tf.shape(Input)[1]

        Input = tf.reshape(Input, [Nsamps*NTbins, xDim], name='X_input')
        
        rangeY = tf.get_variable('rangeY', initializer=params.initrange_Goutmean)
        initSigma = params.initrange_Goutvar
        init_b = params.initbias_Goutmean
        obs_nodes = 64
        fully_connected_layer = FullLayer()
        with tf.variable_scope("obs_nn_mean", reuse=tf.AUTO_REUSE):
            if params.is_identity_output:
                MuY_NTxD = rangeY*Input[:,:yDim]
            elif params.is_linear_output:
                MuY_NTxD = fully_connected_layer(Input, yDim, 'linear', scope='output')
            else:
                full1 = fully_connected_layer(Input, obs_nodes, 'softplus', 'full1')
                full2 = fully_connected_layer(full1, obs_nodes, 'softplus', 'full2')
                MuY_NTxD = fully_connected_layer(full2, yDim, 'linear', 'output',
    #                          initializer=tf.random_uniform_initializer(-rangeY, rangeY),
                                    b_initializer=tf.random_normal_initializer(init_b) )
            MuY_NxTxD = tf.reshape(MuY_NTxD, [Nsamps, NTbins, yDim], name='outY')
        with tf.variable_scope("obs_var", reuse=tf.AUTO_REUSE):
            SigmaInvChol_DxD = tf.get_variable('SigmaInvChol', 
                                initializer=tf.cast(initSigma*tf.eye(yDim), DTYPE))
            self.SigmaChol_1x1xDxD = tf.reshape(tf.matrix_inverse(SigmaInvChol_DxD),
                                      [1, 1, yDim, yDim]) # Needed only for sampling
            SigmaInv_DxD = tf.matmul(SigmaInvChol_DxD, SigmaInvChol_DxD,
                                        transpose_b=True)
            
        return MuY_NxTxD, SigmaInv_DxD 
        
    def compute_LogDensity(self, InputX=None, with_inflow=False):
        """
        Compute the LogDensity
        
        Returns:
            A pair containing:
            
            - LD: The total LogDensity for the Gaussian Generative Model with
                Gaussian Observations.
            - checks: Individual terms forming the latent density. This is useful
                to micromanage the training algorithm, catch difficult bugs, and
                generally better understand what's going on.
        """
        latm = self.lat_ev_model
        X_NxTxd = self.X if InputX is None else InputX
        Nsamps = tf.shape(X_NxTxd)[0]
        NTbins = tf.shape(X_NxTxd)[1]
        if InputX is None:
            LX, checks_LX = latm.logdensity_Xterms, latm.checks_LX # checks = [LX0, LX1, LX2, LX3, LX4]
            MuY_NxTxD, SigmaInvY_DxD = self.MuY_NxTxD, self.SigmaInvY_DxD
        else:
            LX, checks_LX = latm.compute_LogDensity_Xterms(X_NxTxd,
                                                  with_inflow=with_inflow)
            MuY_NxTxD, SigmaInvY_DxD = self._define_mean_variance(X_NxTxd)
        yDim = self.yDim
        
        SigmaInvY_NTxDxD = tf.tile(tf.expand_dims(SigmaInvY_DxD, axis=0),
                                        [Nsamps*NTbins, 1, 1])
        MuY_NTx1xD = tf.reshape(MuY_NxTxD, [Nsamps*NTbins, 1, yDim])
        Y_NTx1xD = tf.reshape(self.Y, [Nsamps*NTbins, 1, yDim])
        
        DeltaY_NTx1xD = Y_NTx1xD - MuY_NTx1xD
        
        LY1 = -0.5*tf.reduce_sum(DeltaY_NTx1xD*tf.matmul(DeltaY_NTx1xD,
                                              SigmaInvY_NTxDxD))
        LY2 = 0.5*(tf.reduce_sum(tf.log(tf.matrix_determinant(SigmaInvY_DxD)))
                            *tf.cast(Nsamps*NTbins, DTYPE))
        LY = tf.add(LY1, LY2, name='LY')
        
        checks = [LY, LX, LY1, LY2]
        checks.extend(checks_LX)
        
        LD = tf.add(LX, LY, name='LogDensity')
        return LD, checks

    def sample_XY(self, sess, Xvar_name='VAEC/X:0', Nsamps=50, 
                  NTbins=100, X0data=None, with_inflow=True, 
                  path_mse_threshold=1.0, draw_plots=False,
                  init_variables=False):
        """
        Sample from the Generative Model for the complete data (X, Y)
        
        Returns:
            A pair containing:
          
            - Ydata_NxTxD: Synthetic observation data 
            - Xdata_NxTxd: Synthetic latent space data
        """
        yDim = self.yDim
        if init_variables:
            sess.run(tf.global_variables_initializer())
            
        Xdata_NxTxd = self.lat_ev_model.sample_X(sess, Xvar_name, 
                                                 Nsamps=Nsamps,
                                                 NTbins=NTbins,
                                                 X0data=X0data,
                                                 with_inflow=with_inflow, 
                                                 path_mse_threshold=path_mse_threshold, 
                                                 draw_plots=draw_plots,
                                                 init_variables=init_variables)
        
        MuY_NxTxD = self.MuY_NxTxD
        SigmaChol_NxTxDxD = tf.tile(self.SigmaChol_1x1xDxD, [Nsamps, NTbins, 1, 1])
        noise_NxTx1xD = tf.random_normal([Nsamps, NTbins, 1, yDim])
        
        scaled_noise_NxTx1xD = tf.matmul(noise_NxTx1xD, SigmaChol_NxTxDxD)
        sampleY_NxTxD = (MuY_NxTxD 
                         + tf.reshape(scaled_noise_NxTx1xD, [Nsamps, NTbins, yDim]))
        Ydata_NxTxD = sess.run(sampleY_NxTxD, feed_dict={Xvar_name : Xdata_NxTxd})
        
        return Ydata_NxTxD, Xdata_NxTxd