from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer: 
    :param names: 
    :param values: 
    :param global_step: 
    :return: 
    """
    for name, value in zip(names, values):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def adj_tensor_dot(adj, y):
    """ Computes the matrix multiplication for the adjacency matrix and the 3D dense matrix y.
    :param adj: square matrix with shape(n_node, n_node)
    :param y: 3D tensor, with shape (batch_size, n_node, output_dim)
    """
    y_shape = [i.value for i in y.shape]
    if len(y_shape) != 3:
        raise Exception('Dimension of y must be 3, instead of: %d' % len(y_shape))

    y_permute_dim = list(range(len(y_shape)))
    y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
    yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
    if isinstance(adj, tf.SparseTensor):
        res = tf.sparse_tensor_dense_matmul(adj, yt)
    else:
        res = tf.matmul(adj, yt)
    res = tf.reshape(res, [y_shape[-2], -1, y_shape[-1]])
    res = tf.transpose(res, perm=[1, 0, 2])
    return res


def dot(x, y):
    """
    Wrapper for tf.matmul for x with rank >= 2.
    :param x: matrix with rank >=2
    :param y: matrix with rank==2
    :return:
    """
    [input_dim, output_dim] = y.get_shape().as_list()

    input_shape = tf.shape(x)
    batch_rank = input_shape.get_shape()[0].value - 1
    batch_shape = input_shape[:batch_rank]
    output_shape = tf.concat(0, [batch_shape, [output_dim]])

    x = tf.reshape(x, [-1, input_dim])
    result_ = tf.matmul(x, y)

    result = tf.reshape(result_, output_shape)

    return result


def get_total_trainable_parameter_size():
    """
    Calculates the total number of trainable parameters in the current graph.
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        total_parameters += np.product([x.value for x in variable.get_shape()])
    return total_parameters


def sparse_matrix_to_tf_sparse_tensor(sparse_mx):
    """Converts sparse matrix to tuple representation as required by tf.SparseTensor"""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        indices = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return indices, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
