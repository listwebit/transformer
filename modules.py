# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Building blocks for Transformer
'''

import numpy as np
import tensorflow as tf

# laye 数据标准化，
def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        # inputs (2, 28, 512)
        # [[[-2.954033    1.695996   -3.1413105  ... -4.018251   -1.3023977
        #    -0.30223972]
        #   [-2.3530521   1.7192748  -3.351814   ... -2.8273482  -0.65232337
        #    -0.4264419 ]
        #   [-2.2563033   1.5593072  -3.2775855  ... -3.2531385  -0.9977432
        #    -0.35023227]
        #   ...
        #   [-0.6843967  -1.003483    1.6845378  ... -0.31852347  2.5649729
        #     0.804183  ]
        #   [-0.9206941  -1.1214577   2.0042856  ...  0.09208354  3.1693528
        #     1.0978277 ]
        #   [-0.74338186 -0.9217373   1.9481943  ...  0.17723289  3.5957296
        #     1.2593844 ]]
        #
        #  [[-2.517493    0.91683435 -3.4142447  ... -3.9844556  -1.1842881
        #    -0.72232884]
        #   [-1.8562554   1.1425755  -3.2166462  ... -4.044012   -1.2189282
        #    -0.38615695]
        #   [-2.0858154   1.4967486  -2.6159432  ... -3.5880928  -0.77991307
        #    -0.33104044]
        inputs_shape = inputs.get_shape() # (2, 28, 512)
        params_shape = inputs_shape[-1:]
        # mean:每个字的均值:(N,L,1):batch=2 (2, 22, 1)
        # [[[-0.06484328]
        #   [-0.01978662]
        #   [-0.05463864]
        #   [-0.03227896]
        #   [-0.05496355]
        #   [ 0.00383393]
        #   [-0.00831935]
        #   [-0.04848529]
        #   [-0.05807229]
        #   [-0.02759192]
        #   [-0.04658262]
        #   [-0.044351  ]
        #   [-0.0404046 ]
        #   [-0.01897928]
        #   [-0.03630898]
        #   [-0.04527371]
        #   [ 0.00215435]
        #   [-0.02535543]
        #   [-0.02192712]
        #   [-0.03002348]
        #   [-0.01454192]
        #   [-0.02426025]]
        #
        #  [[-0.01858747]
        #   [-0.09324092]
        # variance：每个字的方差：(N,L,1):batch=2 (2, 22, 1)
        # [[[10.156523 ]
        #   [ 6.320237 ]
        #   [ 7.1246476]
        #   [ 7.285763 ]
        #   [ 5.4343634]
        #   [ 7.8283257]
        #   [ 6.818144 ]
        #   [ 7.768257 ]
        #   [ 7.3524065]
        #   [ 6.89916  ]
        #   [ 6.85674  ]
        #   [ 7.8054104]
        #   [ 7.141796 ]
        #   [ 6.5306134]
        #   [ 5.97371  ]
        #   [ 6.9241476]
        #   [ 6.4381695]
        #   [ 7.5288787]
        #   [ 7.336317 ]
        #   [ 6.9332967]
        #   [ 5.7453003]
        #   [ 6.994274 ]]
        #
        #  [[ 5.858997 ]
        #   [ 6.8444324]
        #   [ 4.2397   ]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        # beta shape: (512,) 都是0附近的
        # [ 6.42933836e-03 -7.10692781e-04  3.21562425e-03  3.74931638e-04
        #  -4.57212422e-03 -2.78607651e-04 -1.94830136e-05 -2.29233573e-03
        #  -3.16770235e-03 -6.85446896e-04  1.31265656e-03  3.29568808e-04
        #  -3.31255049e-03  2.30989186e-03 -4.35887882e-03 -1.96260633e-04
        #  -9.28307883e-03  5.05952770e-03 -4.32743737e-03  2.50508357e-02
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        #  gamma shape: (512,) 都是1附近的
        # [0.9971061  1.0393478  0.99570525 1.0020051  1.0036978  0.9996209
        #  1.0018373  1.0109453  1.012793   0.9991118  1.0150126  1.0089304
        #  1.0066382  1.0279335  1.0096924  1.0351689  0.9996498  1.0024785
        #  0.988923   1.0238222  1.0073771  0.99466056 1.0291134  1.0062896
        #  0.99483615 1.0175365  1.0021777  0.9909473  1.0064633  1.0024587
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        # normalized :(2, 28, 512) 和上面的inputs是同一个数据
        # [[[-1.5576326   0.8367065  -1.6540633  ... -2.1056075  -0.7071917
        #    -0.19220194]
        #   [-1.1740364   0.8412268  -1.6682914  ... -1.4087503  -0.33240056
        #    -0.22061913]
        #   [-1.2856865   0.8827096  -1.8660771  ... -1.8521839  -0.5704518
        #    -0.20247398]
        #   ...
        #   [-0.36244592 -0.5411275   0.9641068  ... -0.15756473  1.4571316
        #     0.47112694]
        #   [-0.4678502  -0.57143646  1.0413258  ...  0.05470374  1.6424552
        #     0.57362866]
        #   [-0.3773605  -0.47287214  1.064013   ...  0.11564048  1.9462892
        #     0.69514644]]

        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        # 最终outpus:

        outputs = gamma * normalized + beta
        
    return outputs

#构造字向量矩阵
def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings

def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k]. 这个地方错了(h*N, T_q, d_model/h)
    K: Packed keys. 3d tensor. [N, T_k, d_k]. 这个地方错了(h*N, T_q, d_model/h)
    V: Packed values. 3d tensor. [N, T_k, d_v]. 这个地方错了(h*N, T_q, d_model/h)
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product:点积：Q和K的点积
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)


        # scale
        outputs /= d_k ** 0.5

        # key masking
        #  # key_masks:(N,T2)
        #             # [[False False False False False False False False False False  True  True
        #             #   True  True  True  True  True  True  True  True  True  True  True  True
        #             #   True  True  True  True]
        #             #   [False False False False False False False False False  True  True  True
        #             #   True  True  True  True  True  True  True  True  True  True  True  True
        #             #   True  True  True  True]
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        # tf.matmul：矩阵相乘
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"

    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        # tf.tile:平铺；The output tensor's i'th dimension has input.dims(i) * multiples[i] elements,
        # 因为Input进行了多头分割，所有标签也要扩展
        #  batch=2的时候，key_masks:(16, 34)
        #
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
        # tf.expand_dims：Returns a tensor with an additional dimension inserted at index axis.
        # batch=2的时候，key_masks: (16, 1, 34),
        #[[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
        #   [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)

        outputs = inputs + key_masks * padding_num
    # elif type in ("q", "query", "queries"):
    #     # Generate masks
    #     masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    #     masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    #     masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
    #
    #     # Apply masks to inputs
    #     outputs = inputs*masks
    elif type in ("f", "future", "right"):
        # tf.ones_like：Creates a tensor of all ones that has the same shape as the input.
        # diag_vals shape: (20, 20)
        # [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        # tril：生成一个下三角 shape: (20, 20)
        # [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.]
        #  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        #batch取得2，head=8则： future_masks: (16, 20, 20),其中20是每句话词的个数，每一句话生成一个下三角
        # [[[1. 0. 0. ... 0. 0. 0.]
        #   [1. 1. 0. ... 0. 0. 0.]
        #   [1. 1. 1. ... 0. 0. 0.]
        #   ...
        #   [1. 1. 1. ... 1. 0. 0.]
        #   [1. 1. 1. ... 1. 1. 0.]
        #   [1. 1. 1. ... 1. 1. 1.]]
        #
        #  [[1. 0. 0. ... 0. 0. 0.]
        #   [1. 1. 0. ... 0. 0. 0.]
        #   [1. 1. 1. ... 0. 0. 0.]
        #   ...
        #   [1. 1. 1. ... 1. 0. 0.]
        #   [1. 1. 1. ... 1. 1. 0.]
        #   [1. 1. 1. ... 1. 1. 1.]]
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)
        # paddings:按照 paddings 的shap生成一个填充矩阵
        # [[[-4.2949673e+09 -4.2949673e+09 -4.2949673e+09 ... -4.2949673e+09
        #    -4.2949673e+09 -4.2949673e+09]
        #   [-4.2949673e+09 -4.2949673e+09 -4.2949673e+09 ... -4.2949673e+09
        #    -4.2949673e+09 -4.2949673e+09]
        #   [-4.2949673e+09 -4.2949673e+09 -4.2949673e+09 ... -4.2949673e+09
        #    -4.2949673e+09 -4.2949673e+09]
        #   ...
        #   [-4.2949673e+09 -4.2949673e+09 -4.2949673e+09 ... -4.2949673e+09
        #    -4.2949673e+09 -4.2949673e+09]
        #   [-4.2949673e+09 -4.2949673e+09 -4.2949673e+09 ... -4.2949673e+09
        #    -4.2949673e+09 -4.2949673e+09]
        #   [-4.2949673e+09 -4.2949673e+09 -4.2949673e+09 ... -4.2949673e+09
        #    -4.2949673e+09 -4.2949673e+09]]
        paddings = tf.ones_like(future_masks) * padding_num
        # tf.where(condition,paddings, inputs) 按照：condition=true的部位，将padding 填充到input中
        # [[[ 4.43939781e+01 -4.29496730e+09 -4.29496730e+09 ... -4.29496730e+09
        #    -4.29496730e+09 -4.29496730e+09]
        #   [ 3.41685486e+01 -1.06252086e+00 -4.29496730e+09 ... -4.29496730e+09
        #    -4.29496730e+09 -4.29496730e+09]
        #   [ 3.96921425e+01  2.78434825e+00  4.21420937e+01 ... -4.29496730e+09
        #    -4.29496730e+09 -4.29496730e+09]
        #   ...
        #   [ 5.53764381e+01  2.47523355e+00  5.91777534e+01 ... -1.21471321e+02
        #    -4.29496730e+09 -4.29496730e+09]
        #   [ 5.32606239e+01  4.99608231e+00  5.68281059e+01 ... -1.08563034e+02
        #    -4.29496730e+09 -4.29496730e+09]
        #   [ 4.16031532e+01  1.54854858e+00  4.41170464e+01 ... -8.69959717e+01
        #    -4.29496730e+09 -4.29496730e+09]]
        #
        #  [[-1.77314243e+01 -4.29496730e+09 -4.29496730e+09 ... -4.29496730e+09
        #    -4.29496730e+09 -4.29496730e+09]
        #   [-1.86583366e+01 -1.04054117e+01 -4.29496730e+09 ... -4.29496730e+09
        #    -4.29496730e+09 -4.29496730e+09]
        #   [-2.13199158e+01 -1.20762053e+01 -6.59975433e+01 ... -4.29496730e+09
        #    -4.29496730e+09 -4.29496730e+09]
        #   ...
        #   [-2.05263672e+01 -1.20537119e+01 -6.91874313e+01 ... -8.82377319e+01
        #    -4.29496730e+09 -4.29496730e+09]
        #   [-1.61425571e+01 -8.80572414e+00 -5.09573174e+01 ... -6.37179871e+01
        #     1.55839844e+01 -4.29496730e+09]
        #   [-2.01059513e+01 -1.21201067e+01 -6.66899109e+01 ... -8.49668579e+01
        #     2.01012688e+01 -1.04354706e+02]]
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs

# 多头注意力机制
def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        # 先加一层
        Q = tf.layers.dense(queries, d_model, use_bias=True) # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True) # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True) # (N, T_k, d_model)
        
        # Split and concat
        # 切割与合并,这里体现的多头注意力机制
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        # 这里是重点：Attention计算
        # 传入的 src_masks或者 tgt_masks的值
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        # 这里将多头注意力结果恢复N
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = ln(outputs)
 
    return outputs

def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        # inputs
        # outputs： (2, 21, 2048)
        # [[[0.         0.         0.         ... 6.132974   0.         0.        ]
        #   [0.         0.         0.         ... 7.307194   0.         0.        ]
        #   [0.         0.         0.         ... 7.982828   0.         0.        ]
        #   ...
        #   [0.         0.         0.         ... 7.964889   0.         0.        ]
        #   [0.         0.         0.         ... 5.4325767  0.         0.        ]
        #   [0.         0.         0.         ... 7.275622   0.         0.        ]]
        #
        #  [[0.         0.         0.         ... 7.5862374  0.         0.        ]
        #   [0.         0.         0.         ... 5.2648916  0.         0.        ]
        #   [0.         0.23242699 0.         ... 6.187096   0.         0.        ]
        #   ...
        #   [0.         0.         0.         ... 8.233661   0.         0.        ]
        #   [0.         1.2058965  0.         ... 7.485455   0.         0.        ]
        #   [0.         0.         0.         ... 8.30237    0.         0.        ]]]

        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        # outputs (2, 21, 512)
        # [[[-25.37711      1.3692019  -50.80832    ...   0.9118566  -24.534058
        #    -29.246645  ]
        #   [-28.563698     4.2556305  -48.531235   ...  -0.37409064 -25.03862
        #    -28.988926  ]
        #   [-35.380463     9.315982   -49.559715   ...  -2.7038898  -30.601368
        #    -33.075985  ]
        #   ...
        #   [-35.643818     9.848609   -48.638973   ...  -3.1406584  -30.439838
        #    -32.913677  ]
        #   [-21.56209     -0.767092   -48.07494    ...   3.6515896  -22.752474
        #    -24.482904  ]
        #   [-33.58284      8.678018   -46.875854   ...  -2.9922912  -28.830814
        #    -31.506397  ]]
        #
        #  [[-33.383442     9.613742   -46.971367   ...  -3.2966337  -28.788752
        #    -32.695858  ]
        #   [-19.326656    -3.1725218  -50.551548   ...   3.6544466  -20.554945
        #    -25.621178  ]
        #   [-30.288332    17.503975   -24.142347   ...  -9.287842   -23.152536
        #    -29.06433   ]
        #   ...
        #   [-35.35156      9.312039   -49.94821    ...  -2.5631473  -30.52241
        #    -33.088287  ]
        #   [-37.59673     21.043293   -29.895994   ... -10.287443   -30.355291
        #    -35.290684  ]
        #   [-35.986378     9.922715   -50.2701     ...  -3.6265292  -30.689642
        #    -34.769424  ]]]
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = ln(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)

#位置向量
def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

# 改变学习率算法：学习率衰减
def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)