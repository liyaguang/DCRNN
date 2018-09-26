import argparse
import os
import pandas as pd
import sys
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    graph_pkl_filename = 'data/sensor_graph/adj_mx.pkl'
    with open(args.config_filename) as f:
        config = yaml.load(f)
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
        supervisor.restore(sess, config=config)
        df_preds = supervisor.test_and_write_result(sess, config['global_step'])
        # TODO (yaguang): save this file to the npz file.
        for horizon_i in df_preds:
            df_pred = df_preds[horizon_i]
            filename = os.path.join('data/results/', 'dcrnn_prediction_%d.h5' % (horizon_i + 1))
            df_pred.to_hdf(filename, 'results')
        print('Predictions saved as data/results/dcrnn_seq2seq_prediction_[1-12].h5...')


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--traffic_df_filename', default='data/df_highway_2012_4mon_sample.h5',
                        type=str, help='Traffic data file.')
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default=None, type=str, help='Config file for pretrained model.')
    args = parser.parse_args()
    run_dcrnn(args)
