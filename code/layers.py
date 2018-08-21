# Copyright 2017 Daniel Hernandez Diaz, Columbia University
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
import tensorflow as tf

if __name__ == 'layers':
    from utils import variable_in_cpu  # @UnresolvedImport @UnusedImport
else:
    from .utils import variable_in_cpu  # @Reimport

DTYPE = tf.float32

class FullLayer():
    """
    """
    def __init__(self, collections=None):
        """
        """
        self.collections = collections
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax, 'relu' : tf.nn.relu,
                        'sigmoid' : tf.nn.sigmoid}
        
    def __call__(self, Input, nodes, nl='softplus', scope=None, name='out',
                 initializer=tf.orthogonal_initializer(),
                 b_initializer=tf.zeros_initializer()):
        """
        """
        nonlinearity = self.nl_dict[nl]
        input_dim = Input.get_shape()[-1]
        
        if self.collections:
            self.collections += [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES]
            
        with tf.variable_scope(scope or 'fullL'):
            self.weights = weights = variable_in_cpu('weights', [input_dim, nodes], 
                                                     initializer=initializer,
                                                     collections=self.collections)
            biases = variable_in_cpu('biases', [nodes],
                                     initializer=b_initializer,
                                     collections=self.collections)
            full = nonlinearity(tf.matmul(Input, weights) + biases,
                                name=name)
                    
        return full
    
    def get_weights(self):
        return self.weights
    
    
class BatchNormalizationLayer():
    """
    """
    def __init__(self, collections=None):
        """
        """
        self.collections = collections
        self.nl_dict = {'softplus' : tf.nn.softplus, 'linear' : tf.identity,
                        'softmax' : tf.nn.softmax, 'relu' : tf.nn.relu,
                        'lkyrelu' : lambda x : tf.maximum(x, 0.1*x)}
    
    def __call__(self, Input, momentum=0.9, eps=1e-5, scope=None, nl='relu'):
        """
        """
        nonlinearity = self.nl_dict[nl]
        with tf.variable_scope(scope or 'bnL'):
            bn = nonlinearity(tf.contrib.layers.batch_norm(Input, decay=momentum, epsilon=eps,
                                                           scale=True,
                                                           variables_collections=self.collections) )
            return tf.identity(bn, name='batch_norm')