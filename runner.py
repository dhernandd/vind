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

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from code.LatEvModels import LocallyLinearEvolution
from code.ObservationModels import PoissonObs, GaussianObs
from code.Optimizer_VAEC import Optimizer_TS
from code.datetools import addDateTime

# pylint: disable=bad-indentation, no-member, protected-access

DTYPE = tf.float32

# CONFIGURATION
RUN_MODE = 'train' # ['train', 'generate', 'other']

# DIRECTORIES, SAVE FILES, ETC
LOCAL_ROOT = "./"
LOCAL_DATA_DIR = "./data/" 
THIS_DATA_DIR = "gaussian/"
LOCAL_RLT_DIR = "rslts/"
RESTORE_FROM_CKPT = False
LOAD_CKPT_DIR = ""
SAVE_DATA_FILE = "datadict"
SAVE_TO_PY2 = False
IS_PY2 = False

# MODEL/OPTIMIZER ATTRIBUTES
OPT_CLASS = 'ts' # ['ts']
LAT_MOD_CLASS = 'llinear' # ['llinear', 'llwparams']
GEN_MOD_CLASS = 'Gaussian' # ['Gaussian', 'Poisson']
REC_MOD_CLASS = 'SmoothLl' # ['SmoothLl']
YDIM = 18
XDIM = 2
WITH_IDS = False
PDIM = 1
NUM_DIFF_ENTITIES = 1
WITH_INPUTS = False
WITH_MOD_DYNAMICS = False
WITH_ITERM = False
INCLUDE_WITH_INPUTS = False
IDIM = 1
NNODES = 70
ALPHA = 0.1
INITRANGE_MUX = 1.0
INITRANGE_LAMBDAX = 0.1
INITRANGE_B = 1.0
POISSON_INITRANGE_OUTY = 5.0
INIT_Q0 = 0.4
INIT_Q = 1.0
IS_Q_TRAINABLE = True
INITRANGE_GOUTMEAN = 9.0
INITRANGE_GOUTVAR = 1.0
INITBIAS_GOUTMEAN = 1.0
POISSON_IS_OUT_POSITIVE = False
POISSON_INV_TAU = 0.2
IS_LINEAR_OUTPUT = False
IS_IDENTITY_OUTPUT = False
PLOT2D = True

# TRAINING PARAMETERS
LEARNING_RATE = 1e-3
END_LR = 1e-4
NUM_FPIS = 2
USE_GRAD_TERM = False
USE_TRANSPOSE_TRICK = True
NUM_EPS_TO_INCLUDE_GRADS = 2000
BATCH_SIZE = 1
NUM_EPOCHS = 500
SHUFFLE = True
EPOCHS_TO_INCLUDE_INPUTS = 50
NUM_GRAD_STEPS = 1

# GENERATION PARAMETERS
NTBINS = 30
NSAMPS = 200
DRAW_HEAT_MAPS = False

flags = tf.app.flags #@UndefinedVariable
flags.DEFINE_string('mode', RUN_MODE, "The mode in which to run. Can be ['train', 'generate']")

flags.DEFINE_string('local_root', LOCAL_ROOT, "The root directory of VIND.")
flags.DEFINE_string('local_data_dir', LOCAL_DATA_DIR, "The directory that stores all "
                                                       "the datasets") 
flags.DEFINE_string('local_rlt_dir', LOCAL_RLT_DIR, "The directory that stores the results")
flags.DEFINE_string('this_data_dir', THIS_DATA_DIR, "For the 'generate' mode, the directory "
                                                    "that shall store this dataset")
flags.DEFINE_string('save_data_file', SAVE_DATA_FILE, "For the 'generate' mode, the name "
                    "of the file to store the data")
flags.DEFINE_boolean('restore_from_ckpt', RESTORE_FROM_CKPT, "Shall VIND restore a "
                                                "previously trained model?")
flags.DEFINE_string('load_ckpt_dir', LOAD_CKPT_DIR, "For the 'train' mode, the directory "
                    "that will store tensorflow's checkpoints.")
flags.DEFINE_boolean('save_to_py2', SAVE_TO_PY2, "Should the data be pickled in a Python 2 "
                                                   "compatible protocol?")
flags.DEFINE_boolean('is_py2', IS_PY2, "Was the data pickled in python 2?")

flags.DEFINE_integer('xDim', XDIM, "The dimensionality of the latent space")
flags.DEFINE_integer('yDim', YDIM, "The dimensionality of the data")

flags.DEFINE_string('lat_mod_class', LAT_MOD_CLASS, "A string denoting the evolution "
                    "model class. One of ['llinear']")
flags.DEFINE_float('alpha', ALPHA, "Key hyperparameter. Controls the scale of the "
                   "nonlinearity. Increasing ALPHA reduces the a priori smoothness "
                   "of the flow. This parameters works in conjunction with initrange_B")
flags.DEFINE_float('initrange_B', INITRANGE_B, "Key hyperparameter. Controls the initial "
                   "size of the nonlinearity. Bigger values for INITRANGE_B lead to less "
                   "smooth paths. Works in conjunction with alpha")
flags.DEFINE_float('init_Q0', INIT_Q0, "Key hyperparameter. Controls the initial spread "
                   "of the starting points of the paths in latent space. This parameter "
                   "should be adjusted to work efficiently with INITRANGE_MUX so that the "
                   "starting points of the inferred latent paths initially cover the "
                   "full bounding box.")
flags.DEFINE_float('init_Q', INIT_Q, "Controls the initial noise added to the paths "
                   "in latent space. More importantly, it also controls the initial "
                   "ranges within which the latent space paths are contained. Roughly "
                   "rangeX ~ 1/(Lambda + Q), so if Q is very big, the range is reduced. "
                   "If Q is very small, then it defers to Lambda. Optimally "
                   "Lambda ~ Q ~ 1.")
flags.DEFINE_boolean('is_Q_trainable', IS_Q_TRAINABLE, "Should the variance of the "
                     "Evolution Model be trained?")

flags.DEFINE_string('rec_mod_class', REC_MOD_CLASS, "A string denoting the recognition "
                    "model class. Implemented ['SmoothLl']")
flags.DEFINE_float('initrange_MuX', INITRANGE_MUX, "Key hyperparameter. Controls the "
                   "initial ranges in the latent space, within which the latent paths "
                   "are contained. Bigger values for INITRANGE_MUX lead to bigger "
                   "bounding box. This parameters should be adjusted initially so that "
                   "the initial paths do not collapse nor blow up.")
flags.DEFINE_float('initrange_LambdaX', INITRANGE_LAMBDAX, "Gives further controls for "
                   "the initial ranges within which the latent space paths are "
                   "contained. Roughly rangeX ~ 1/(Lambda + Q), so for larger Lambda "
                   "the range is reduced. If Lambda is small, then control depends only "
                   "on Q. Optimally Lambda ~ Q ~ 1.")

flags.DEFINE_string('gen_mod_class', GEN_MOD_CLASS, "A string denoting the generative "
                    "model class. One of ['Poisson', 'Gaussian']")
flags.DEFINE_float('initrange_Goutmean', INITRANGE_GOUTMEAN, "Controls the scale of the "
                   "initial output of the Generative Network for Gaussian observations. ")
flags.DEFINE_float('initrange_Goutvar', INITRANGE_GOUTVAR, "Controls the scale of the "
                   "variance of the Generative Network outputs for Gaussian observations. ")
flags.DEFINE_float('initbias_Goutmean', INITBIAS_GOUTMEAN, "Controls the bias added to "
                   "Generative Network in the case of Gaussian observations")
flags.DEFINE_float('initrange_outY', POISSON_INITRANGE_OUTY, "Controls the initial "
                   "range of the output of the Poisson generative network")
flags.DEFINE_boolean('poisson_is_out_positive', POISSON_IS_OUT_POSITIVE, "")
flags.DEFINE_float('poisson_inv_tau', POISSON_INV_TAU, "")
flags.DEFINE_boolean('is_linear_output', IS_LINEAR_OUTPUT, "")
flags.DEFINE_boolean('is_identity_output', IS_IDENTITY_OUTPUT, "")

flags.DEFINE_boolean('with_ids', WITH_IDS, "Does the data include known Ids for the "
                     "different trials (e.g. cell types)")
flags.DEFINE_integer('num_diff_entities', NUM_DIFF_ENTITIES, "Number of different Ids"
                     "present in this data")
flags.DEFINE_integer('pDim', PDIM, "Dimension of the subspace in the latent space "
                     "associated to the Id information of the trial (as opposed to "
                     "the state information)")

flags.DEFINE_boolean('with_inputs', WITH_INPUTS, "Does the data include known inputs "
                     "for each time point?")
flags.DEFINE_integer('iDim', IDIM, "Dimensionality of the inputs")
flags.DEFINE_integer('epochs_to_include_inputs', EPOCHS_TO_INCLUDE_INPUTS, "Number of "
                     "epochs in which the model is trained with data that does not "
                     "include inputs before the data with inputs is added")
flags.DEFINE_boolean('with_mod_dynamics', WITH_MOD_DYNAMICS, "")
flags.DEFINE_boolean('with_Iterm', WITH_ITERM, "")
flags.DEFINE_boolean('include_with_inputs', INCLUDE_WITH_INPUTS, "")

flags.DEFINE_integer('num_grad_steps', NUM_GRAD_STEPS, "")
flags.DEFINE_boolean('plot2d', PLOT2D, "")


flags.DEFINE_string('opt_class', OPT_CLASS, "A string denoting the optimizer class. "
                    "One of ['ts']")
flags.DEFINE_float('learning_rate', LEARNING_RATE, "It's the starting learning rate")
flags.DEFINE_float('end_lr', END_LR, "Final value for a learning rate that is "
                   "scheduled to decrease at an exponential rate")
flags.DEFINE_integer('num_fpis', NUM_FPIS, "Number of Fixed-Point Iterations to "
                     "carry per epoch. The bigger this value, the slower the "
                     "algorithm. However, it may happen, specially at the beginning "
                     "of training, that setting this value > 1 leads to better "
                     "results. ")
flags.DEFINE_boolean('use_grad_term', USE_GRAD_TERM, "Should I include the term with "
                     "gradients in the posterior formula? Discarding them justified "
                     "almost always since the term tends to be subleading. Moreover, "
                     "setting this to False leads to a SIGNIFICANT speed up because "
                     "computing these gradients is the costliest operation timewise. "
                     "`USE_GRAD_TERM=False` is nonetheless an additional "
                     "approximation. Use carefully.")
flags.DEFINE_boolean('use_transpose_trick', USE_TRANSPOSE_TRICK, "Enforce A = A.T for "
                     "the evolution model (this 'trick' has proven to be very beneficial "
                     "for training, leading in general to better fits)")
flags.DEFINE_integer('num_eps_to_include_grads', NUM_EPS_TO_INCLUDE_GRADS, "Number "
                     "of epochs after which the exact gradient terms should be "
                     "included in the computation of the posterior.")
flags.DEFINE_integer('batch_size', BATCH_SIZE, "You guessed it.")
flags.DEFINE_integer('num_epochs', NUM_EPOCHS, "Number of training epochs.")
flags.DEFINE_boolean('shuffle', SHUFFLE, "Should I shuffle the data before starting "
                     "a new epoch?")

flags.DEFINE_integer('genNsamps', NSAMPS, "The number of samples to generate")
flags.DEFINE_integer('genNTbins', NTBINS, "The number of time bins in the generated "
                     "data")
flags.DEFINE_boolean('draw_heat_maps', DRAW_HEAT_MAPS, "Should I draw heat maps of "
                     "your data?")

params = flags.FLAGS


def write_option_file(path):
    """
    Write a file "params.txt" with the parameters used for this fit.
    
    Args:
        path (str): The path to the folder where the file is stored
    """
    params_list = sorted([param for param in dir(params) if param 
                          not in ['h', 'help', 'helpfull', 'helpshort']])
    with open(path + 'params.txt', 'w') as option_file:
        for par in params_list:
            option_file.write(par + ' ' + str(getattr(params, par)) + '\n')
                
                
def generate_fake_data(pars, data_path=None, save_data_file=None,
                       draw_heat_maps=True, savefigs=True):
    """
    Generate synthetic data and possibly pickle it for later use.
    
    Args:
        pars (dict):
        data_path: The local directory where the generated data should be stored.
        save_data_file: The name of the file to hold your data
        Nsamps: Number of trials to generate
        NTbins: Number of time steps to run.
        xDim: The dimensions of the latent space.
        yDim: The dimensions of the data.
        write_params_file: Would you like the parameters with which this data has been 
                    generated to be saved to a separate txt file?
    """    
    print('Generating some fake data...!\n')
    lat_mod_classes = {'llinear' : LocallyLinearEvolution}
    gen_mod_classes = {'Poisson' : PoissonObs, 'Gaussian' : GaussianObs}

    lat_mod_class, gen_mod_class = pars.lat_mod_class, pars.gen_mod_class
    evolution_class = lat_mod_classes[lat_mod_class]
    generator_class = gen_mod_classes[gen_mod_class]

    if data_path:
        if not type(save_data_file) is str:
            raise ValueError("`save_data_file` must be string (the name of the file) "
                             "if you intend to save the data (`data_path` is not None)")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        else:
            print("The data directory {} ".format(data_path), " already exists. "
                  "You may be overwriting a dataset in it, "
                  "Would you like to proceed? (y/n)")
            a = input()
            if a == 'n':
                raise Exception("Please change the value of the global "
                                "variable THIS_DATA_DIR")
            elif a != 'y':
                raise Exception("Please type 'y' or 'n'")            
        write_option_file(data_path)
    
    # Generate some fake data for training, validation and test
    Nsamps=pars.genNsamps
    NTbins=pars.genNTbins
    xDim = pars.xDim
    yDim = pars.yDim
    with tf.Session() as sess:
        X = tf.placeholder(DTYPE, shape=[None, None, xDim], name='X')
        Y = tf.placeholder(DTYPE, shape=[None, None, yDim], name='Y')
        if lat_mod_class in ['llwparams']:
            Ids = tf.placeholder(tf.int32, [None], name='Ids')
            latm = evolution_class(X, Ids, pars)
        else:
            latm = evolution_class(X, pars)
        genm = generator_class(Y, X, pars, latm)
    
        Nsamps_train = int(4*Nsamps/5)
        valid_test = int(Nsamps/10)
        sess.run(tf.global_variables_initializer())
        data = genm.sample_XY(sess, Xvar_name='X:0', Nsamps=Nsamps,
                              NTbins=NTbins, with_inflow=True)
        print("Done.")
        
        Ydata, Xdata = data[0], data[1]
        Ytrain, Xtrain = Ydata[:Nsamps_train], Xdata[:Nsamps_train]
        Yvalid, Xvalid = Ydata[Nsamps_train:-valid_test], Xdata[Nsamps_train:-valid_test]
        Ytest, Xtest = Ydata[valid_test:], Xdata[valid_test:]
        if lat_mod_class in ['llwparams']:
            Iddata = data[2]
            Idtrain = Iddata[:Nsamps_train]
            Idvalid, Idtest = Iddata[Nsamps_train:-valid_test], Iddata[valid_test:]
        
        # If xDim == 2, draw a cool path plot
        if xDim == 2:
            if lat_mod_class in ['llwparams']:
                for ent in range(pars.num_diff_entities):
                    print('Plottins DS for entity ', str(ent), '...')
                    list_idxs = [i for i, Id in enumerate(Iddata) if Id == ent]
                    XdataId = Xdata[list_idxs]
                    latm.plot_2Dquiver_paths(sess, XdataId, [ent], rlt_dir=data_path,
                                             rslt_file='quiver_plot_'+str(ent),
                                             with_inflow=True, savefig=savefigs)
            else:
                latm.plot_2Dquiver_paths(sess, Xdata, Xvar_name='X:0', rlt_dir=data_path,
                                         with_inflow=True, savefig=savefigs)
        if draw_heat_maps:
            maxY = np.max(Ydata)
            for i in range(1):
                plt.figure()
                sns.heatmap(Ydata[i].T, yticklabels=False, vmax=maxY).get_figure()
                if savefigs:
                    plt.savefig(data_path + "heat" + str(i) + ".png")
                else:
                    plt.show()
                    plt.pause(0.001)
                    input('Press Enter to continue.')
                    plt.close()
            
    if data_path:
        datadict = {'Ytrain' : Ytrain, 'Yvalid' : Yvalid,
                    'Xtrain' : Xtrain, 'Xvalid' : Xvalid,
                    'Ytest' : Ytest, 'Xtest' : Xtest}
        if lat_mod_class in ['llwparams']:
            datadict.update({'Idtrain' : Idtrain, 'Idvalid' : Idvalid,
                             'Idtest' : Idtest})
        with open(data_path + save_data_file, 'wb+') as data_file:
            pickle.dump(datadict, data_file)
    
        if pars.save_to_py2:
            with open(data_path + save_data_file + '_py2', 'wb+') as data_file:
                pickle.dump(datadict, data_file, protocol=2)
            
    return Ydata, Xdata

def build(pars, rlt_dir):
    """
    Build a VIND model that stores results into rlt_dir
    """    
    if not os.path.exists(rlt_dir):
        os.makedirs(rlt_dir)
        write_option_file(rlt_dir)
    
    opt_classes = {'ts' : Optimizer_TS}
    Optimizer_class = opt_classes[pars.opt_class]
    opt = Optimizer_class(pars)
    return opt
                
def train(pars, data_path, rlt_dir):
    """
    Train a VIND model, possibly from a saved checkpoint
    """
    with open(data_path + pars.save_data_file, 'rb+') as f:
        # Set encoding='latin1' for python 2 pickled data
        datadict = pickle.load(f, encoding='latin1') if pars.is_py2 else pickle.load(f)
    pars.yDim = datadict['Ytrain'].shape[-1]
    if not bool(pars.alpha) and pars.use_transpose_trick:
        print("You cannot use the transpose trick when fitting global linear dynamics. "
              "Setting use_transpose_trick to False.")
        pars.use_transpose_trick = False

    opt = build(pars, rlt_dir)
    sess = tf.get_default_session()
    with sess:
        if pars.restore_from_ckpt:
            saver = opt.saver
            print("Restoring from ", pars.load_ckpt_dir, " ...\n")
            ckpt_state = tf.train.get_checkpoint_state(pars.load_ckpt_dir)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            print("Done.")
        else:
            sess.run(tf.global_variables_initializer())
        opt.train(sess, rlt_dir, datadict, num_epochs=pars.num_epochs)
        
def other(datadict, data_path):
    """
    Temporary function to do one time things
    """
    datadict['Ytrain_wI_whole'] = datadict['Ytrain_wI'] 
    datadict['Yvalid_wI_whole'] = datadict['Yvalid_wI'] 
    datadict['Ytest_wI_whole'] = datadict['Ytest_wI'] 
    datadict['Ytrain_wI'] = datadict['Ytrain_wI_whole'][:,:,0:1]
    datadict['Yvalid_wI'] = datadict['Yvalid_wI_whole'][:,:,0:1]
    datadict['Ytest_wI'] = datadict['Ytest_wI_whole'][:,:,0:1]
    with open(data_path+params.save_data_file, 'wb+') as f:
        pickle.dump(datadict, f)

def main(_):
    """
    Fly babe!
    """
    data_path = params.local_data_dir + params.this_data_dir
    rlt_dir = ( params.local_rlt_dir + params.this_data_dir + addDateTime() + '/'
                if not params.restore_from_ckpt else
                params.load_ckpt_dir )
    if params.mode == 'generate':
        generate_fake_data(params, data_path=data_path,
                           save_data_file=params.save_data_file)
    elif params.mode == 'train':
        sess = tf.Session()
        with sess.as_default():   #pylint: disable=not-context-manager
            train(params, data_path, rlt_dir)
    elif params.mode == 'other':
        with open(data_path+params.save_data_file, 'rb+') as f:
            # Set encoding='latin1' for python 2 pickled data
            datadict = pickle.load(f, encoding='latin1') if params.is_py2 else pickle.load(f)
            print(sorted(datadict.keys()))
        other(datadict, data_path)
        
    
if __name__ == '__main__':
    tf.app.run()
    
    from sys import platform
    if platform == 'darwin':
        os.system('say "There is a beer in your fridge"')