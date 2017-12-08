import unittest

import numpy as np
import pandas as pd

from lib import utils
from lib.utils import StandardScaler


class MyTestCase(unittest.TestCase):
    def test_separate_seasonal_trend_and_residual(self):
        data = np.array([
            [2, 1, 2, 3, 0, 1, 2, 1, 2, 3, 4, 3, 0, 3, 4, 1]
        ], dtype=np.float32).T
        trends = np.array([
            [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2]
        ], dtype=np.float32).T
        residual = np.array([
            [1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1]
        ], dtype=np.float32).T
        df = pd.DataFrame(data)
        df_trend, df_residual = utils.separate_seasonal_trend_and_residual(df, period=4, test_ratio=0, null_val=-1)
        self.assertTrue(np.array_equal(df_trend.values, trends))
        self.assertTrue(np.array_equal(df_residual.values, residual))

    def test_get_rush_hours_bool_index(self):
        index = pd.date_range('2017-02-27', '2017-03-06', freq='1min')
        data = np.zeros((len(index), 3))
        df = pd.DataFrame(data, index=index)
        ind = utils.get_rush_hours_bool_index(df)
        df = df[ind]
        self.assertEqual(6 * 5 * 60, df.shape[0])


class IODataPreparationTest(unittest.TestCase):
    from lib import utils
    def test_generate_io_data_with_time(self):
        data = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ], dtype=np.float32).T
        df = pd.DataFrame(data, index=pd.date_range('2017-10-18', '2017-10-19 23:59', freq='3h'))
        xs, ys = utils.generate_io_data_with_time(df, batch_size=2, seq_len=3, horizon=3, output_type='range', )
        self.assertTupleEqual(xs.shape, (3, 2, 9))
        self.assertTupleEqual(ys.shape, (3, 2, 6))


class StandardScalerTest(unittest.TestCase):
    def test_transform(self):
        data = np.array([
            [35., 0.],
            [0., 17.5],
            [70., 35.]]
        )
        expected_result = np.array([
            [0., -1.],
            [-1, -0.5],
            [1., 0.]]
        )
        scaler = StandardScaler(mean=35., std=35.)
        result = scaler.transform(data)
        self.assertTrue(np.array_equal(expected_result, result))

    def test_transform_df(self):
        df = pd.DataFrame([
            [35., 0.],
            [0., 17.5],
            [70., 35.]]
        )
        expected_result = np.array([
            [0., -1.],
            [-1, -0.5],
            [1., 0.]]
        )
        scaler = StandardScaler(mean=35., std=35.)
        result = scaler.transform(df)

        self.assertTrue(np.array_equal(expected_result, result.values))

    def test_reverse_transform(self):
        data = np.array([
            [0., -1.],
            [-1, -0.5],
            [1., 0.]]
        )
        expected_result = np.array([
            [35., 0.],
            [0., 17.5],
            [70., 35.]]
        )
        scaler = StandardScaler(mean=35., std=35.)
        result = scaler.inverse_transform(data)
        self.assertTrue(np.array_equal(expected_result, result))

    def test_reverse_transform_df(self):
        df = pd.DataFrame([
            [0., -1.],
            [-1, -0.5],
            [1., 0.]]
        )
        expected_result = np.array([
            [35., 0.],
            [0., 17.5],
            [70., 35.]]
        )
        scaler = StandardScaler(mean=35., std=35.)
        result = scaler.inverse_transform(df)
        self.assertTrue(np.array_equal(expected_result, result.values))


if __name__ == '__main__':
    unittest.main()
