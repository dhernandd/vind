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
import numpy as np

import tensorflow as tf

DTYPE = tf.float32

def variable_in_cpu(name, shape, initializer, collections=None):
    """
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, dtype=DTYPE, initializer=initializer,
                              collections=collections)
    return var


def blk_tridiag_chol(A_Txdxd, B_Tm1xdxd):
    """
    Compute the Cholesky decomposition of a symmetric, positive definite
    block-tridiagonal matrix.

    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix 
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper) 1st block 
        off-diagonal matrix

    Outputs: 
    R - python list with two elements
        * R[0] - [T x n x n] tensor of block diagonal elements of Cholesky decomposition
        * R[1] - [T-1 x n x n] tensor of (lower) 1st block off-diagonal elements of Cholesky
    """
    def compute_chol(LC, AB_2xdxd):
        L_dxd = LC[0]
        A_dxd, B_dxd = AB_2xdxd[0], AB_2xdxd[1]
        C_dxd = tf.matmul(B_dxd, tf.matrix_inverse(L_dxd), 
                      transpose_a=True, transpose_b=True)
        D = A_dxd - tf.matmul(C_dxd, C_dxd, transpose_b=True)
        L_dxd = tf.cholesky(D)
        return [L_dxd, C_dxd]
        
    L1_dxd = tf.cholesky(A_Txdxd[0])
    C1_dxd = tf.zeros_like(B_Tm1xdxd[0], dtype=DTYPE)
    
    result_2xTm1xdxd = tf.scan(fn=compute_chol, elems=[A_Txdxd[1:], B_Tm1xdxd],
                               initializer=[L1_dxd, C1_dxd])

    AChol_Txdxd = tf.concat([tf.expand_dims(L1_dxd, 0), result_2xTm1xdxd[0]], 
                            axis=0)    
    BChol_Tm1xdxd = result_2xTm1xdxd[1]
    
    return [AChol_Txdxd, BChol_Tm1xdxd]


def blk_chol_inv(A_Txdxd, B_Tm1xdxd, b_Txd, lower=True, transpose=False):
    """
    Solve the equation Cx = b for x, where C is assumed to be a block-bi-
    diagonal triangular matrix - only the first lower/upper off-diagonal block
    is nonvanishing.
    
    This function will be used to solve the equation Mx = b where M is a
    block-tridiagonal matrix due to the fact that M = C^T*C where C is block-
    bidiagonal triangular.
    
    Inputs:
    A - [T x n x n]   tensor, where each A[i,:,:] is the ith block diagonal matrix 
    B - [T-1 x n x n] tensor, where each B[i,:,:] is the ith (upper or lower) 
        1st block off-diagonal matrix
     
    lower (default: True) - boolean specifying whether to treat B as the lower
          or upper 1st block off-diagonal of matrix C
    transpose (default: False) - boolean specifying whether to transpose the 
          off-diagonal blocks B[i,:,:] (useful if you want to compute solve 
          the problem C^T x = b with a representation of C.) 
 
    Outputs: 
    x - solution of Cx = b
    """
    # Define a matrix-vector dot product because the tensorflow developers feel
    # this is beneath them.
    tf_dot = lambda M, v : tf.reduce_sum(tf.multiply(M, v), axis=1)
    if transpose:
        A_Txdxd = tf.transpose(A_Txdxd, [0,2,1])
        B_Tm1xdxd = tf.transpose(B_Tm1xdxd, [0,2,1])
    
    # Whether B is lower or upper doesn't matter. The function to be passed to
    # scan is the same.
    def step(x_d, ABb_2x_):
        A_dxd, B_dxd, b_d = ABb_2x_[0], ABb_2x_[1], ABb_2x_[2]
        return tf_dot(tf.matrix_inverse(A_dxd),
                         b_d - tf_dot(B_dxd, x_d))
    if lower:
        x0_d = tf_dot(tf.matrix_inverse(A_Txdxd[0]), b_Txd[0])
        result_Tm1xd = tf.scan(fn=step, elems=[A_Txdxd[1:], B_Tm1xdxd, b_Txd[1:]], 
                             initializer=x0_d)
        result_Txd = tf.concat([tf.expand_dims(x0_d, axis=0), result_Tm1xd], axis=0)
    else:
        xN_d = tf_dot(tf.matrix_inverse(A_Txdxd[-1]), b_Txd[-1])
        result_Tm1xd = tf.scan(fn=step, 
                             elems=[A_Txdxd[:-1][::-1], B_Tm1xdxd[::-1], b_Txd[:-1][::-1]],
                             initializer=xN_d )
        result_Txd = tf.concat([tf.expand_dims(xN_d, axis=0), result_Tm1xd],
                               axis=0)[::-1]

    return result_Txd 




if __name__ == '__main__':
    # Test `blk_tridiag_chol`
    
    # Note that the matrices forming the As here are symmetric. Also, I have
    # chosen the entries wisely - cuz I'm a wise guy - so that the overall
    # matrix is positive definite as required by the algo.
    mA = np.mat('1  0.2; 0.2 7', dtype='f')
    mB = np.mat('3.0  0.0; 0.0 1.0', dtype='f')
    mC = np.mat('2  0.4; 0.4 3', dtype='f')
    mD = np.mat('3  0.8; 0.8 1', dtype='f')

    mE = np.mat('0.02 0.07; 0.01 0.04', dtype='f')
    mF = np.mat('0.07  0.02; 0.09 0.03', dtype='f')
    mZ = np.mat('0.0 0.0; 0.0 0.0', dtype='f')
    
    mat = np.bmat([[mA, mE, mZ, mZ],
                   [mE.T, mB, mF, mZ],
                   [mZ, mF.T, mC, mE],
                   [mZ, mZ, mE.T, mD]])
    
    tA = tf.get_variable('tA', initializer=mA, dtype=DTYPE)
    tB = tf.get_variable('tB', initializer=mB, dtype=DTYPE)
    tC = tf.get_variable('tC', initializer=mC, dtype=DTYPE)
    tD = tf.get_variable('tD', initializer=mD, dtype=DTYPE)

    tE = tf.get_variable('tE', initializer=mE, dtype=DTYPE)
    tF = tf.get_variable('tF', initializer=mF, dtype=DTYPE)

    As = tf.stack([tA, tB, tC, tD])
    Bs = tf.stack([tE, tF, tE])
    
    AChol, BChol = blk_tridiag_chol(As, Bs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        AChol, BChol = sess.run([AChol, BChol])
        print('Len (AChol, BChol):', len(AChol), len(BChol))
        
        Chol_mat = np.bmat([[AChol[0], mZ, mZ, mZ],
                            [BChol[0], AChol[1], mZ, mZ],
                            [mZ, BChol[1], AChol[2], mZ],
                            [mZ, mZ, BChol[2], AChol[3]]])
        new_mat = np.dot(Chol_mat, Chol_mat.T)
        print( ("Sing, goddess! Ain't it TRUE that we have found the Cholesky decomposition of"
                " a block-tridiagonal matrix?... \nThat is"),
              np.allclose(mat, new_mat))
        print('')
        
    # Test `blk_chol_inv` Before I used `mA`, `mB`, etc to construct a matrix to
    # for which I found the Cholesky decomposition. Now I use the same matrices
    # to construct the Cholesky decomposition of some matrix.
    lowermat = np.bmat([[mA,     mZ, mZ,   mZ],
                        [mE.T,   mB, mZ,   mZ],
                        [mZ,   mF.T, mC,   mZ],
                        [mZ,     mZ, mE.T, mD]])
    lwrmat_sq = np.dot(lowermat, lowermat.T)
    
    # To find the Cholesky inverse, we pass the LOWER blocks of the blocks-
    # tridiagonal matrix
    Bs = tf.stack([tf.transpose(tE), tf.transpose(tF), tf.transpose(tE)])
    
    # Initialize a numpy vector
    vb = np.mat('1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0', dtype='f')
    tb = tf.get_variable('b', initializer=vb, dtype=DTYPE)
     
    # Apply `blk_chol_inv` to `vb` twice
    temp_ib = blk_chol_inv(As, Bs, vb)
    res = blk_chol_inv(As, Bs, temp_ib, lower=False, transpose=True)
 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(res)
        true_res = np.linalg.inv(lwrmat_sq).dot(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
        print( ("...and, ain't it TRUE that we have found the solution x to Mx = b"
                " via x = CC^Tb \nwhere C is the Cholesky decomposition of block-tridiagonal M?"
                "\nMmmm, that is"), np.allclose(res.flatten(), true_res))
