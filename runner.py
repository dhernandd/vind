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
INITRANGE_OUTY = 5.0
INIT_Q0 = 0.4
INIT_Q = 1.0
IS_Q_TRAINABLE = True
INITRANGE_GOUTMEAN = 9.0
INITRANGE_GOUTVAR = 1.0
INITBIAS_GOUTMEAN = 1.0
IS_OUT_POSITIVE = False
IS_LINEAR_OUTPUT = False
IS_IDENTITY_OUTPUT = False
INV_TAU = 0.2
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
flags.DEFINE_string('local_data_dir', LOCAL_DATA_DIR, ("The directory that stores all "
                                                       "the datasets") )
flags.DEFINE_string('local_rlt_dir', LOCAL_RLT_DIR, "The directory that stores the results")
flags.DEFINE_string('this_data_dir', THIS_DATA_DIR, ("For the 'generate' mode, the directory "
                                                     "that shall store this dataset"))
flags.DEFINE_string('save_data_file', SAVE_DATA_FILE, ("For the 'generate' mode, the name of the file "
                                                       "to store the data"))
flags.DEFINE_boolean('restore_from_ckpt', RESTORE_FROM_CKPT, ("Should VIND restore a "
                                                "previously trained model?") )
flags.DEFINE_string('load_ckpt_dir', LOAD_CKPT_DIR, ("For the 'train' mode, the directory storing "
                                                       "`tf` checkpoints."))
flags.DEFINE_boolean('save_to_py2', SAVE_TO_PY2, ("Should the data be pickled in a Python 2 "
                                                   "compatible protocol?") )
flags.DEFINE_boolean('is_py2', IS_PY2, "Was the data pickled in python 2?")

flags.DEFINE_string('opt_class', OPT_CLASS, ("The optimizer class. Implemented ['struct', 'ts']"))
flags.DEFINE_string('lat_mod_class', LAT_MOD_CLASS, ("The evolution model class. Implemented "
                                                     "['llinear']"))
flags.DEFINE_string('gen_mod_class', GEN_MOD_CLASS, ("The generative model class. Implemented "
                                                     "['Poisson, Gaussian']"))
flags.DEFINE_string('rec_mod_class', REC_MOD_CLASS, ("The recognition model class. Implemented "
                                                     "['SmoothLl']"))
flags.DEFINE_integer('xDim', XDIM, "The dimensionality of the latent space")
flags.DEFINE_integer('yDim', YDIM, "The dimensionality of the data")
flags.DEFINE_float('alpha', ALPHA, ("The scale factor of the nonlinearity. This parameters "
                                    "works in conjunction with initrange_B"))
flags.DEFINE_float('initrange_MuX', INITRANGE_MUX, ("Controls the initial ranges within "
                                           "which the latent space paths are contained. Bigger "
                                           "values here lead to bigger bounding box. It is im-"
                                           "portant to adjust this parameter so that the initial "
                                           "paths do not collapse nor blow up."))
flags.DEFINE_float('initrange_LambdaX', INITRANGE_LAMBDAX, ("Controls the initial ranges within "
                                                "which the latent space paths are contained. Roughly "
                                                "rangeX ~ 1/(Lambda + Q), so if Lambda very big, the "
                                                "range is reduced. If Lambda very small, then it defers "
                                                "to Q. Optimally Lambda ~ Q ~ 1."))
flags.DEFINE_float('initrange_B', INITRANGE_B, ("Controls the initial size of the nonlinearity. "
                                                "Works in conjunction with alpha"))
flags.DEFINE_float('initrange_outY', INITRANGE_OUTY, ("Controls the initial range of the output of the "
                                                "generative network"))
flags.DEFINE_float('init_Q0', INIT_Q0, ("Controls the initial spread of the starting points of the "
                                    "paths in latent space."))
flags.DEFINE_float('init_Q', INIT_Q, ("Controls the initial noise added to the paths in latent space. "
                                      "More importantly, it also controls the initial ranges within "
                                      "which the latent space paths are contained. Roughly rangeX ~  "
                                      "1/(Lambda + Q), so if Q is very big, the range is reduced. If "
                                      "Q is very small, then it defers to Lambda. Optimally "
                                      "Lambda ~ Q ~ 1."))
flags.DEFINE_float('initrange_Goutmean', INITRANGE_GOUTMEAN, ("Controls the scale of the initial output "
                                                              "of the Generative Network for Gaussian "
                                                              "observations. "))
flags.DEFINE_float('initrange_Goutvar', INITRANGE_GOUTVAR, "")
flags.DEFINE_float('initbias_Goutmean', INITBIAS_GOUTMEAN, "")
flags.DEFINE_float('inv_tau', INV_TAU, "")
flags.DEFINE_boolean('is_Q_trainable', IS_Q_TRAINABLE, "")
flags.DEFINE_boolean('is_out_positive', IS_OUT_POSITIVE, "")
flags.DEFINE_boolean('is_linear_output', IS_LINEAR_OUTPUT, "")
flags.DEFINE_boolean('is_identity_output', IS_IDENTITY_OUTPUT, "")
flags.DEFINE_boolean('with_ids', WITH_IDS, "")
flags.DEFINE_integer('num_diff_entities', NUM_DIFF_ENTITIES, "")
flags.DEFINE_integer('pDim', PDIM, "")
flags.DEFINE_boolean('with_inputs', WITH_INPUTS, "")
flags.DEFINE_integer('iDim', IDIM, "")
flags.DEFINE_integer('epochs_to_include_inputs', EPOCHS_TO_INCLUDE_INPUTS, "")
flags.DEFINE_boolean('with_mod_dynamics', WITH_MOD_DYNAMICS, "")
flags.DEFINE_boolean('with_Iterm', WITH_ITERM, "")
flags.DEFINE_boolean('include_with_inputs', INCLUDE_WITH_INPUTS, "")
flags.DEFINE_integer('num_grad_steps', NUM_GRAD_STEPS, "")
flags.DEFINE_boolean('plot2d', PLOT2D, "")


flags.DEFINE_float('learning_rate', LEARNING_RATE, "It's the starting learning rate, silly")
flags.DEFINE_float('end_lr', END_LR, ("For a learning rate that decreases at an exponential "
                                      "rate, this is its final value.") )
flags.DEFINE_integer('num_fpis', NUM_FPIS, ("Number of Fixed-Point Iterations to carry per epoch. "
                                        "The bigger this value, the slower the algorithm. "
                                        "However, it may happen, specially at the beginning of "
                                        "training, that setting this value > 1 leads to better "
                                        "results. "))
flags.DEFINE_boolean('use_grad_term', USE_GRAD_TERM, ("Should I include the term with gradients "
                                        "in the posterior formula? Discarding them is often "
                                        "justified since the term tends to be subleading. "
                                        "Moreover, setting this to False leads to a "
                                        "SIGNIFICANT speed up because computing the grads "
                                        "is the costliest operation timewise. On the other " 
                                        "hand, it IS an approximation. Use carefully.") )
flags.DEFINE_boolean('use_transpose_trick', USE_TRANSPOSE_TRICK, (""))
flags.DEFINE_integer('num_eps_to_include_grads', NUM_EPS_TO_INCLUDE_GRADS, ("Number of epochs "
                                        "after which the exact gradient terms should be "
                                        "included in the computation of the posterior.") )
flags.DEFINE_integer('batch_size', BATCH_SIZE, "You guessed it.")
flags.DEFINE_integer('num_epochs', NUM_EPOCHS, "Number of training epochs.")
flags.DEFINE_boolean('shuffle', SHUFFLE, "Should I shuffle the data before starting a new epoch?")

flags.DEFINE_integer('genNsamps', NSAMPS, "The number of samples to generate")
flags.DEFINE_integer('genNTbins', NTBINS, "The number of time bins in the generated data")
flags.DEFINE_boolean('draw_heat_maps', DRAW_HEAT_MAPS, "Should I draw heat maps of your data?")

params = flags.FLAGS


def write_option_file(path):
    """
    Writes a file with the parameters that were used for this fit. Cuz - no doubt -
    you will forget master.
    """
    params_list = sorted([param for param in dir(params) if param 
                          not in ['h', 'help', 'helpfull', 'helpshort']])
    with open(path + 'params.txt', 'w') as option_file:
        for par in params_list:
            option_file.write(par + ' ' + str(getattr(params, par)) + '\n')
                
def generate_fake_data(params, data_path=None, save_data_file=None,
                       draw_heat_maps=True, savefigs=True):
    """
    Generates synthetic data and possibly pickles it for later use. Maybe you
    would like to train something on it? ;)
    
    Args:
        lat_mod_class: A string that is a key to the evolution model class.
        gen_mod_class: A string that is a key to the observation model class.
        data_path: The local directory where the generated data should be stored. If None,
                    don't store shit.
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

    lat_mod_class, gen_mod_class = params.lat_mod_class, params.gen_mod_class
    evolution_class = lat_mod_classes[lat_mod_class]
    generator_class = gen_mod_classes[gen_mod_class]

    if data_path:
        if not type(save_data_file) is str:
            raise ValueError("`save_data_file` must be string (representing the name of your file) "
                             "if you intend to save the data (`data_path` is not None)")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        else:
            print("This data directory already exists. You may be overwriting a dataset in it, "
                  "Would you like to proceed? (y/n)")
            a = input()
            if a == 'n':
                raise Exception("Then please change the value of the global "
                                "variable THIS_DATA_DIR")
            elif a != 'y':
                raise Exception("I have little patience for a computer program. Make sure "
                                "to type 'n' or 'y' next time")            
        write_option_file(data_path)
    
    # Generate some fake data for training, validation and test
    Nsamps=params.genNsamps
    NTbins=params.genNTbins
    xDim = params.xDim
    yDim = params.yDim
    with tf.Session() as sess:
        X = tf.placeholder(DTYPE, shape=[None, None, xDim], name='X')
        Y = tf.placeholder(DTYPE, shape=[None, None, yDim], name='Y')
        if lat_mod_class in ['llwparams']:
            Ids = tf.placeholder(tf.int32, [None], name='Ids')
            latm = evolution_class(X, Ids, params)
        else:
            latm = evolution_class(X, params)
        genm = generator_class(Y, X, params, latm)
    
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
                for ent in range(params.num_diff_entities):
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
        datadict = {'Ytrain' : Ytrain, 'Yvalid' : Yvalid, 'Xtrain' : Xtrain, 'Xvalid' : Xvalid,
                    'Ytest' : Ytest, 'Xtest' : Xtest}
        if lat_mod_class in ['llwparams']:
            datadict.update({'Idtrain' : Idtrain, 'Idvalid' : Idvalid, 'Idtest' : Idtest})
        with open(data_path + save_data_file, 'wb+') as data_file:
            pickle.dump(datadict, data_file)
    
        if params.save_to_py2:
            with open(data_path + save_data_file + '_py2', 'wb+') as data_file:
                pickle.dump(datadict, data_file, protocol=2)
            
    return Ydata, Xdata

def build(params, rlt_dir):
    """
    Builds a VIND model that stores results into rlt_dir
    """    
    if not os.path.exists(rlt_dir):
        os.makedirs(rlt_dir)
        write_option_file(rlt_dir)
    
    opt_classes = {'ts' : Optimizer_TS}
    Optimizer_class = opt_classes[params.opt_class]
    opt = Optimizer_class(params)
    return opt
                
def train(params, data_path, rlt_dir):
    """
    Trains a VIND model, possibly from a saved checkpoint
    """
    with open(data_path+params.save_data_file, 'rb+') as f:
        # Set encoding='latin1' for python 2 pickled data
        datadict = pickle.load(f, encoding='latin1') if params.is_py2 else pickle.load(f)
    params.yDim = datadict['Ytrain'].shape[-1]
    if not bool(params.alpha) and params.use_transpose_trick:
        print("You cannot use the transpose trick when fitting global linear dynamics. "
              "Setting use_transpose_trick to False.")
        params.use_transpose_trick = False

    opt = build(params, rlt_dir)
    sess = tf.get_default_session()
    with sess:
        if params.restore_from_ckpt:
            saver = opt.saver
            print("Restoring from ", params.load_ckpt_dir, " ...\n")
            ckpt_state = tf.train.get_checkpoint_state(params.load_ckpt_dir)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            print("Done.")
        else:
            sess.run(tf.global_variables_initializer())
        opt.train(sess, rlt_dir, datadict, num_epochs=params.num_epochs)
        
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
        with sess.as_default():
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
