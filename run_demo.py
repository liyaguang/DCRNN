import json
import os
import pandas as pd
import sys
import tensorflow as tf

from lib.dcrnn_utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('use_cpu_only', False, 'Whether to run tensorflow on cpu.')


def run_dcrnn(traffic_reading_df):
    run_id = 'dcrnn_DR_2_h_12_64-64_lr_0.01_bs_64_d_0.00_sl_12_MAE_1207002222'

    log_dir = os.path.join('data/model', run_id)

    config_filename = 'config_100.json'
    graph_pkl_filename = 'data/sensor_graph/adj_mx.pkl'
    with open(os.path.join(log_dir, config_filename)) as f:
        config = json.load(f)
    tf_config = tf.ConfigProto()
    if FLAGS.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(traffic_reading_df, config=config, adj_mx=adj_mx)
        supervisor.restore(sess, config=config)
        df_preds = supervisor.test_and_write_result(sess, config['global_step'])
        for horizon_i in df_preds:
            df_pred = df_preds[horizon_i]
            filename = os.path.join('data/results/', 'dcrnn_prediction_%d.h5' % (horizon_i + 1))
            df_pred.to_hdf(filename, 'results')
        print('Predictions saved as data/results/dcrnn_seq2seq_prediction_[1-12].h5...')


# def run_fc_lstm(traffic_reading_df):
#     run_id = 'fclstm_h_12_256-256_lr_0.0001_bs_64_d_0.00_sl_12_MAE_1026175218'
#     log_dir = os.path.join('data/model', run_id)
#
#     config_filename = 'config_59.json'
#     # graph_pkl_filename = 'data/sensor_graph/sensor_graph_exp.pkl'
#     with open(os.path.join(log_dir, config_filename)) as f:
#         config = json.load(f)
#     tf_config = tf.ConfigProto()
#     if FLAGS.use_cpu_only:
#         tf_config = tf.ConfigProto(device_count={'GPU': 0})
#     tf_config.gpu_options.allow_growth = True
#     # Set small entries to zero for sparseness.
#     with tf.Session(config=tf_config) as sess:
#         supervisor = LSTMSeq2SeqSupervisor(traffic_reading_df, config=config)
#         supervisor.restore(sess, config=config)
#         df_preds = supervisor.test_and_write_result(sess, config['global_step'])
#         for horizon_i in df_preds:
#             df_pred = df_preds[horizon_i]
#             # filename = os.path.join('data/results/', 'gcrnn_seq2seq_prediction_%d.h5' % (horizon_i + 1))
#             filename = os.path.join('data/results/', 'fc_lstm_prediction_%d.h5' % (horizon_i + 1))
#             df_pred.to_hdf(filename, 'results')
#         print('Predictions saved as data/results/fc_lstm_prediction_[1-12].h5...')


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    traffic_df_filename = 'data/df_highway_2012_4mon_sample.h5'
    traffic_reading_df = pd.read_hdf(traffic_df_filename)
    run_dcrnn(traffic_reading_df)
    # run_fc_lstm(traffic_reading_df)
