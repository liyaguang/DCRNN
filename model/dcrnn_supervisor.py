from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import pandas as pd
import tensorflow as tf

from lib.utils import generate_graph_seq2seq_io_data_with_time
from model.dcrnn_model import DCRNNModel
from model.tf_model_supervisor import TFModelSupervisor


class DCRNNSupervisor(TFModelSupervisor):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, traffic_reading_df, adj_mx, config):
        self._adj_mx = adj_mx
        super(DCRNNSupervisor, self).__init__(config, df_data=traffic_reading_df)

    def _prepare_train_val_test_data(self):
        # Parsing model parameters.
        batch_size = self._get_config('batch_size')
        horizon = self._get_config('horizon')
        seq_len = self._get_config('seq_len')

        test_batch_size = 1
        add_time_in_day = self._get_config('add_time_in_day')

        num_nodes = self._df_train.shape[-1]
        x_train, y_train = generate_graph_seq2seq_io_data_with_time(self._df_train,
                                                                    batch_size=batch_size,
                                                                    seq_len=seq_len,
                                                                    horizon=horizon,
                                                                    num_nodes=num_nodes,
                                                                    scaler=self._scaler,
                                                                    add_time_in_day=add_time_in_day,
                                                                    add_day_in_week=False)
        x_val, y_val = generate_graph_seq2seq_io_data_with_time(self._df_val, batch_size=batch_size,
                                                                seq_len=seq_len,
                                                                horizon=horizon,
                                                                num_nodes=num_nodes,
                                                                scaler=self._scaler,
                                                                add_time_in_day=add_time_in_day,
                                                                add_day_in_week=False)
        x_test, y_test = generate_graph_seq2seq_io_data_with_time(self._df_test,
                                                                  batch_size=test_batch_size,
                                                                  seq_len=seq_len,
                                                                  horizon=horizon,
                                                                  num_nodes=num_nodes,
                                                                  scaler=self._scaler,
                                                                  add_time_in_day=add_time_in_day,
                                                                  add_day_in_week=False)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _build_train_val_test_models(self):
        # Builds the model.
        input_dim = self._x_train.shape[-1]
        num_nodes = self._df_test.shape[-1]
        output_dim = self._get_config('output_dim')
        test_batch_size = self._get_config('test_batch_size')
        train_config = dict(self._config)
        train_config.update({
            'input_dim': input_dim,
            'num_nodes': num_nodes,
            'output_dim': output_dim,
        })
        test_config = dict(self._config)
        test_config.update({
            'batch_size': test_batch_size,
            'input_dim': input_dim,
            'num_nodes': num_nodes,
            'output_dim': output_dim,
        })

        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                train_model = DCRNNModel(is_training=True, config=train_config, scaler=self._scaler,
                                         adj_mx=self._adj_mx)

        with tf.name_scope('Val'):
            with tf.variable_scope('DCRNN', reuse=True):
                val_model = DCRNNModel(is_training=False, config=train_config, scaler=self._scaler,
                                       adj_mx=self._adj_mx)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                test_model = DCRNNModel(is_training=False, config=test_config, scaler=self._scaler,
                                        adj_mx=self._adj_mx)

        return train_model, val_model, test_model

    def _convert_model_outputs_to_eval_df(self, y_preds):
        y_preds = np.stack(y_preds, axis=1)
        # y_preds: (batch_size, epoch_size, horizon, num_nodes, output_dim)
        # horizon = y_preds.shape[2]
        horizon = self._get_config('horizon')
        num_nodes = self._df_train.shape[-1]
        df_preds = {}
        for horizon_i in range(horizon):
            y_pred = np.reshape(y_preds[:, :, horizon_i, :, 0], self._eval_dfs[horizon_i].shape)
            df_pred = pd.DataFrame(self._scaler.inverse_transform(y_pred), index=self._eval_dfs[horizon_i].index,
                                   columns=self._eval_dfs[horizon_i].columns)
            df_preds[horizon_i] = df_pred
        return df_preds

    @staticmethod
    def _generate_run_id(config):
        batch_size = config.get('batch_size')
        dropout = config.get('dropout')
        learning_rate = config.get('learning_rate')
        loss_func = config.get('loss_func')
        max_diffusion_step = config['max_diffusion_step']
        num_rnn_layers = config.get('num_rnn_layers')
        rnn_units = config.get('rnn_units')
        seq_len = config.get('seq_len')
        structure = '-'.join(
            ['%d' % rnn_units for _ in range(num_rnn_layers)])
        horizon = config.get('horizon')
        filter_type = config.get('filter_type')
        filter_type_abbr = 'L'
        if filter_type == 'random_walk':
            filter_type_abbr = 'R'
        elif filter_type == 'dual_random_walk':
            filter_type_abbr = 'DR'
        run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_d_%.2f_sl_%d_%s_%s/' % (
            filter_type_abbr, max_diffusion_step, horizon,
            structure, learning_rate, batch_size,
            dropout, seq_len, loss_func,
            time.strftime('%m%d%H%M%S'))
        return run_id
