from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import sys
import tensorflow as tf
import time
import yaml

from lib import utils, log_helper, metrics
from lib.utils import StandardScaler, DataLoader

from model.dcrnn_model import DCRNNModel


class DCRNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, adj_mx, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_helper.config_logging(log_dir=self.log_dir, log_filename='info.log', level=logging.DEBUG)
        self._writer = tf.summary.FileWriter(self._log_dir)
        logging.info(kwargs)

        # Data preparation
        self._data = self._prepare_data(**self._data_kwargs)

        # Build models.
        scaler = self._data['scaler']

        self._epoch = 0
        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                self._train_model = DCRNNModel(is_training=True, scaler=scaler,
                                               batch_size=self._data_kwargs['batch_size'],
                                               adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Val'):
            with tf.variable_scope('DCRNN', reuse=True):
                self._val_model = DCRNNModel(is_training=False, scaler=scaler,
                                             batch_size=self._data_kwargs['batch_size'],
                                             adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                self._test_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=self._data_kwargs['test_batch_size'],
                                              adj_mx=adj_mx, **self._model_kwargs)

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        logging.info('Total number of trainable parameters: %d' % total_trainable_parameter)
        for var in tf.global_variables():
            logging.info('%s, %s' % (var.name, var.get_shape()))

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @staticmethod
    def _prepare_data(dataset_dir, **kwargs):
        data = {}
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y']
        scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
            data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
        for k, v in data.items():
            logging.info((k, v.shape))
        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], kwargs['batch_size'], shuffle=True)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], kwargs['val_batch_size'], shuffle=False)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], kwargs['test_batch_size'], shuffle=False)
        data['scaler'] = scaler

        return data

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def _train(self, sess, base_lr, epoch, steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10, **train_kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        max_to_keep = train_kwargs.get('max_to_keep', 100)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        model_filename = train_kwargs.get('model_filename')
        if model_filename is not None:
            saver.restore(sess, model_filename)
            self._epoch = epoch + 1
        else:
            sess.run(tf.global_variables_initializer())
        while self._epoch <= epochs:
            # Learning rate schedule.
            new_lr = max(min_learning_rate, base_lr * (lr_decay_ratio ** np.sum(self._epoch >= np.array(steps))))
            self._train_model.set_lr(sess=sess, lr=new_lr)
            sys.stdout.flush()

            start_time = time.time()
            train_results = self._train_model.run_epoch_generator(sess, self._train_model,
                                                                  self._data['train_loader'].get_iterator(),
                                                                  train_op=self._train_model.train_op,
                                                                  writer=self._writer)
            train_loss, train_mae = train_results['loss'], train_results['mae']
            if train_loss > 1e5:
                logging.warning('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = self._val_model.run_epoch_generator(sess, self._val_model,
                                                              self._data['val_loader'].get_iterator(),
                                                              train_op=None)
            val_loss, val_mae = val_results['loss'], val_results['mae']

            utils.add_simple_summary(self._writer,
                                        ['loss/train_loss', 'metric/train_mae', 'loss/val_loss', 'metric/val_mae'],
                                        [train_loss, train_mae, val_loss, val_mae], global_step=global_step)
            end_time = time.time()
            message = 'Epoch [{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} lr:{:.6f} {:.1f}s'.format(
                self._epoch, global_step, train_mae, val_mae, new_lr, (end_time - start_time))
            logging.info(message)
            if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
                self.test_and_write_result(sess, global_step)
            if val_loss <= min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save_model(sess, saver, val_loss)
                logging.info(
                    'Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss, val_loss, model_filename))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    logging.warning('Early stopping at epoch: %d' % self._epoch)
                    break

            history.append(val_mae)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
        return np.min(history)

    def test_and_write_result(self, sess, global_step, **kwargs):
        test_results = self._test_model.run_epoch_generator(sess, self._test_model,
                                                            self._data['test_loader'].get_iterator(),
                                                            return_output=True, train_op=None)

        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds = test_results['loss'], test_results['outputs']
        utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)

        y_preds = np.concatenate(y_preds, axis=0)
        scaler = self._data['scaler']
        outputs = []
        for horizon_i in range(self._data['y_test'].shape[1]):
            y_truth = np.concatenate(self._data['y_test'][:, horizon_i, :, 0], axis=0)
            y_truth = scaler.inverse_transform(y_truth)
            y_pred = np.concatenate(y_preds[:, horizon_i, :, 0], axis=0)
            y_pred = y_pred[:y_truth.shape[0], ...]  # Only take the batch number
            y_pred = scaler.inverse_transform(y_pred)
            outputs.append(y_pred)
            mae = metrics.masked_mae_np(y_pred, y_truth, null_val=0)
            mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
            rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
            logging.info(
                "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                    horizon_i + 1, mae, mape, rmse
                )
            )
            utils.add_simple_summary(self._writer,
                                        ['%s_%d' % (item, horizon_i + 1) for item in
                                         ['metric/rmse', 'metric/mape', 'metric/mae']],
                                        [rmse, mape, mae],
                                        global_step=global_step)
        return y_preds

    @staticmethod
    def restore(sess, config):
        """
        Restore from saved model.
        :param sess:
        :param config:
        :return:
        """
        model_filename = config['train'].get('model_filename')
        max_to_keep = config['train'].get('max_to_keep', 100)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        saver.restore(sess, model_filename)

    def save_model(self, sess, saver, val_loss):
        config_filename = 'config_{}.yaml'.format(self._epoch)
        config = dict(self._kwargs)
        global_step = np.asscalar(sess.run(tf.train.get_or_create_global_step()))
        config['train']['epoch'] = self._epoch
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = saver.save(sess,
                                                       os.path.join(self._log_dir, 'models-{:.4f}'.format(val_loss)),
                                                       global_step=global_step, write_meta_graph=False)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']

    @property
    def log_dir(self):
        return self._log_dir
