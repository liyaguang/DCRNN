import argparse
import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.var_model import VAR

from lib import utils
from lib.metrics import masked_rmse_np, masked_mape_np, masked_mae_np
from lib.utils import StandardScaler


def historical_average_predict(df, period=12 * 24 * 7, test_ratio=0.2, null_val=0.):
    """
    Calculates the historical average of sensor reading.
    :param df:
    :param period: default 1 week.
    :param test_ratio:
    :param null_val: default 0.
    :return:
    """
    n_sample, n_sensor = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    y_test = df[-n_test:]
    y_predict = pd.DataFrame.copy(y_test)

    for i in range(n_train, min(n_sample, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]
        y_predict.iloc[i - n_train, :] = historical[historical != null_val].mean()
    # Copy each period.
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict.iloc[start:start + size, :] = y_predict.iloc[start - period: start + size - period, :].values
    return y_predict, y_test


def static_predict(df, n_forward, test_ratio=0.2):
    """
    Assumes $x^{t+1} = x^{t}$
    :param df:
    :param n_forward:
    :param test_ratio:
    :return:
    """
    test_num = int(round(df.shape[0] * test_ratio))
    y_test = df[-test_num:]
    y_predict = df.shift(n_forward).iloc[-test_num:]
    return y_predict, y_test


def var_predict(df, n_forwards=(1, 3), n_lags=4, test_ratio=0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_sample, n_output = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = df[:n_train], df[n_train:]

    scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
    data = scaler.transform(df_train.values)
    var_model = VAR(data)
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    # Do forecasting.
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        prediction = var_result.forecast(scaler.transform(df.values[input_ind: input_ind + n_lags]), max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    df_predicts = []
    for i, n_forward in enumerate(n_forwards):
        df_predict = pd.DataFrame(scaler.inverse_transform(result[i]), index=df_test.index, columns=df_test.columns)
        df_predicts.append(df_predict)
    return df_predicts, df_test


def eval_static(traffic_reading_df):
    logger.info('Static')
    horizons = [1, 3, 6, 12]
    logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in horizons:
        y_predict, y_test = static_predict(traffic_reading_df, n_forward=horizon, test_ratio=0.2)
        rmse = masked_rmse_np(preds=y_predict.as_matrix(), labels=y_test.as_matrix(), null_val=0)
        mape = masked_mape_np(preds=y_predict.as_matrix(), labels=y_test.as_matrix(), null_val=0)
        mae = masked_mae_np(preds=y_predict.as_matrix(), labels=y_test.as_matrix(), null_val=0)
        line = 'Static\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_historical_average(traffic_reading_df, period):
    y_predict, y_test = historical_average_predict(traffic_reading_df, period=period, test_ratio=0.2)
    rmse = masked_rmse_np(preds=y_predict.as_matrix(), labels=y_test.as_matrix(), null_val=0)
    mape = masked_mape_np(preds=y_predict.as_matrix(), labels=y_test.as_matrix(), null_val=0)
    mae = masked_mae_np(preds=y_predict.as_matrix(), labels=y_test.as_matrix(), null_val=0)
    logger.info('Historical Average')
    logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in [1, 3, 6, 12]:
        line = 'HA\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_var(traffic_reading_df, n_lags=3):
    n_forwards = [1, 3, 6, 12]
    y_predicts, y_test = var_predict(traffic_reading_df, n_forwards=n_forwards, n_lags=n_lags,
                                     test_ratio=0.2)
    logger.info('VAR (lag=%d)' % n_lags)
    logger.info('Model\tHorizon\tRMSE\tMAPE\tMAE')
    for i, horizon in enumerate(n_forwards):
        rmse = masked_rmse_np(preds=y_predicts[i].as_matrix(), labels=y_test.as_matrix(), null_val=0)
        mape = masked_mape_np(preds=y_predicts[i].as_matrix(), labels=y_test.as_matrix(), null_val=0)
        mae = masked_mae_np(preds=y_predicts[i].as_matrix(), labels=y_test.as_matrix(), null_val=0)
        line = 'VAR\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def main(args):
    traffic_reading_df = pd.read_hdf(args.traffic_reading_filename)
    eval_static(traffic_reading_df)
    eval_historical_average(traffic_reading_df, period=7 * 24 * 12)
    eval_var(traffic_reading_df, n_lags=3)


if __name__ == '__main__':
    logger = utils.get_logger('data/model', 'Baseline')
    parser = argparse.ArgumentParser()
    parser.add_argument('--traffic_reading_filename', default="data/metr-la.h5", type=str,
                        help='Path to the traffic Dataframe.')
    args = parser.parse_args()
    main(args)
