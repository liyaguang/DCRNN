import unittest

import numpy as np
import tensorflow as tf

from lib import tf_utils


class TensorDotTest(unittest.TestCase):
    def test_adj_tensor_dot(self):
        # adj: [[1, 0], [0, 1]]
        # SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
        adj_indices = [[0, 0], [1, 1]]
        adj_values = np.array([1, 1], dtype=np.float32)
        adj_shape = [2, 2]
        adj = tf.SparseTensor(adj_indices, adj_values, adj_shape)
        # y: (2, 2, 2), [[[1, 0], [0, 1]], [[1, 1], [1, 1]]]
        y = np.array([[[1, 0], [0, 1]], [[1, 1], [1, 1]]], dtype=np.float32)
        y = tf.constant(y)
        expected_result = np.array([[[1, 0], [0, 1]], [[1, 1], [1, 1]]], dtype=np.float32)
        result = tf_utils.adj_tensor_dot(adj, y)
        with tf.Session() as sess:
            result_ = sess.run(result)
            self.assertTrue(np.array_equal(expected_result, result_))


if __name__ == '__main__':
    unittest.main()
