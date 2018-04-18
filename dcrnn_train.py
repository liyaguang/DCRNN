from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import yaml


from lib import log_helper
from lib.dcrnn_utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor

# flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', -1, 'Batch size')
flags.DEFINE_integer('cl_decay_steps', -1,
                     'Parameter to control the decay speed of probability of feeding groundth instead of model output.')
flags.DEFINE_string('config_filename', None, 'Configuration filename for restoring the model.')
flags.DEFINE_integer('epochs', -1, 'Maximum number of epochs to train.')
flags.DEFINE_string('filter_type', None, 'laplacian/random_walk/dual_random_walk.')
flags.DEFINE_string('graph_pkl_filename', 'data/sensor_graph/adj_mx.pkl',
                    'Pickle file containing: sensor_ids, sensor_id_to_ind_map, dist_matrix')
flags.DEFINE_integer('horizon', -1, 'Maximum number of timestamps to prediction.')
flags.DEFINE_float('l1_decay', -1.0, 'L1 Regularization')
flags.DEFINE_float('lr_decay', -1.0, 'Learning rate decay.')
flags.DEFINE_integer('lr_decay_epoch', -1, 'The epoch that starting decaying the parameter.')
flags.DEFINE_integer('lr_decay_interval', -1, 'Interval beteween each deacy.')
flags.DEFINE_float('learning_rate', -1, 'Learning rate. -1: select by hyperopt tuning.')
flags.DEFINE_string('log_dir', None, 'Log directory for restoring the model from a checkpoint.')
flags.DEFINE_string('loss_func', None, 'MSE/MAPE/RMSE_MAPE: loss function.')
flags.DEFINE_float('min_learning_rate', -1, 'Minimum learning rate')
flags.DEFINE_integer('nb_weeks', 17, 'How many week\'s data should be used for train/test.')
flags.DEFINE_integer('patience', -1,
                     'Maximum number of epochs allowed for non-improving validation error before early stopping.')
flags.DEFINE_integer('seq_len', -1, 'Sequence length.')
flags.DEFINE_integer('test_every_n_epochs', -1, 'Run model on the testing dataset every n epochs.')
flags.DEFINE_string('traffic_df_filename', 'data/df_highway_2012_4mon_sample.h5',
                    'Path to hdf5 pandas.DataFrame.')
flags.DEFINE_bool('use_cpu_only', False, 'Set to true to only use cpu.')
flags.DEFINE_bool('use_curriculum_learning', None, 'Set to true to use Curriculum learning in decoding stage.')
flags.DEFINE_integer('verbose', -1, '1: to log individual sensor information.')


def main():
    # Reads graph data.
    with open(FLAGS.config_filename) as f:
        supervisor_config = yaml.load(f)
        logger = log_helper.get_logger(supervisor_config.get('base_dir'), 'info.log')
        logger.info('Loading graph from: ' + FLAGS.graph_pkl_filename)
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(FLAGS.graph_pkl_filename)
        adj_mx[adj_mx < 0.1] = 0
        logger.info('Loading traffic data from: ' + FLAGS.traffic_df_filename)
        traffic_df_filename = FLAGS.traffic_df_filename
        traffic_reading_df = pd.read_hdf(traffic_df_filename)
        traffic_reading_df = traffic_reading_df.ix[:, sensor_ids]
        supervisor_config['use_cpu_only'] = FLAGS.use_cpu_only
        if FLAGS.log_dir:
            supervisor_config['log_dir'] = FLAGS.log_dir
        if FLAGS.use_curriculum_learning is not None:
            supervisor_config['use_curriculum_learning'] = FLAGS.use_curriculum_learning
        if FLAGS.loss_func:
            supervisor_config['loss_func'] = FLAGS.loss_func
        if FLAGS.filter_type:
            supervisor_config['filter_type'] = FLAGS.filter_type
        # Overwrites space with specified parameters.
        for name in ['batch_size', 'cl_decay_steps', 'epochs', 'horizon', 'learning_rate', 'l1_decay',
                     'lr_decay', 'lr_decay_epoch', 'lr_decay_interval', 'learning_rate', 'min_learning_rate',
                     'patience', 'seq_len', 'test_every_n_epochs', 'verbose']:
            if getattr(FLAGS, name) >= 0:
                supervisor_config[name] = getattr(FLAGS, name)

        tf_config = tf.ConfigProto()
        if FLAGS.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(traffic_reading_df=traffic_reading_df, adj_mx=adj_mx,
                                         config=supervisor_config)

            supervisor.train(sess=sess)


if __name__ == '__main__':
    main()
