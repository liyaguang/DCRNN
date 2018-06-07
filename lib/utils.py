import datetime
import numpy as np
import pandas as pd
import pickle


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def get_rush_hours_bool_index(df, hours=((7, 10), (17, 20)), weekdays=(0, 5)):
    """
    Calculates predator of rush hours: 7:00am - 9:59am,  4:00pm-7:59am, Mon-Fri.
    :param df:
    :param hours: a tuple of two, (start_hour, end_hour)
    :param weekdays: a tuple of two, (start_weekday, end_weekday)
    """
    # Week day.
    weekday_predate = (df.index.dayofweek >= weekdays[0]) & (df.index.dayofweek < weekdays[1])
    # Hours.
    hour_predate = (df.index.time >= datetime.time(hours[0][0], 0)) & (df.index.time < datetime.time(hours[0][1], 0))
    hour_predate |= (df.index.time >= datetime.time(hours[1][0], 0)) & (df.index.time < datetime.time(hours[1][1], 0))

    return weekday_predate & hour_predate


def generate_io_data(data, seq_len, horizon=1, scaler=None):
    """
    Generates input, output data which are
    Args:
        :param data: tensor
        :param seq_len: length of the sequence, or timesteps.
        :param horizon: the horizon of prediction.
        :param strides:
        :param scaler:
        :return  (X, Y) i.e., input, output
    """
    xs, ys = [], []
    total_seq_len, _ = data.shape
    assert np.ndim(data) == 2
    if scaler:
        data = scaler.transform(data)
    for i in range(0, total_seq_len - horizon - seq_len + 1):
        x_i = data[i: i + seq_len, :]
        y_i = data[i + seq_len + horizon - 1, :]
        xs.append(x_i)
        ys.append(y_i)
    xs = np.stack(xs, axis=0)
    ys = np.stack(ys, axis=0)
    return xs, ys


def generate_io_data_with_time(df, batch_size, seq_len, horizon, output_type='point', scaler=None,
                               add_time_in_day=True, add_day_in_week=False):
    """

    :param df:
    :param batch_size:
    :param seq_len:
    :param horizon:
    :param output_type: point, range, seq2seq
    :param scaler:
    :param add_time_in_day:
    :param add_day_in_week:
    :return:
    x, y, both are 3-D tensors with size (epoch_size, batch_size, input_dim).
    """
    if scaler:
        df = scaler.transform(df)
    num_samples, num_nodes = df.shape
    data = df.values
    batch_len = num_samples // batch_size
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        data_list.append(time_ind.reshape(-1, 1))
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, 7))
        day_in_week[np.arange(num_samples), df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    data = data[:batch_size * batch_len, :].reshape((batch_size, batch_len, -1))
    xs, ys = [], []
    for i in range(seq_len, batch_len - horizon + 1):
        x_i, y_i = None, None
        if output_type == 'point':
            x_i = data[:, i - seq_len: i, :].reshape((batch_size, -1))
            y_i = data[:, i + horizon - 1, :num_nodes].reshape((batch_size, -1))
        elif output_type == 'range':
            x_i = data[:, i - seq_len: i, :].reshape((batch_size, -1))
            y_i = data[:, i: i + horizon, :num_nodes].reshape((batch_size, -1))
        elif output_type == 'seq2seq':
            x_i = data[:, i - seq_len: i, :]
            y_i = data[:, i: i + horizon, :]
        xs.append(x_i)
        ys.append(y_i)
    xs = np.stack(xs, axis=0)
    ys = np.stack(ys, axis=0)
    return xs, ys


def generate_graph_seq2seq_io_data_with_time(df, batch_size, seq_len, horizon, num_nodes, scaler=None,
                                             add_time_in_day=True, add_day_in_week=False):
    """

    :param df: 
    :param batch_size: 
    :param seq_len: 
    :param horizon: 
    :param scaler: 
    :param add_day_in_week:
    :return: 
    x, y, both are 5-D tensors with size (epoch_size, batch_size, seq_len, num_sensors, input_dim).
    Adjacent batches are continuous sequence, i.e., x[i, j, :, :] is before x[i+1, j, :, :]
    """
    if scaler:
        df = scaler.transform(df)
    num_samples, _ = df.shape
    data = df.values
    batch_len = num_samples // batch_size
    data = np.expand_dims(data, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    data = data[:batch_size * batch_len, :, :].reshape((batch_size, batch_len, num_nodes, -1))
    epoch_size = batch_len - seq_len - horizon + 1
    x, y = [], []
    for i in range(epoch_size):
        x_i = data[:, i: i + seq_len, ...]
        y_i = data[:, i + seq_len: i + seq_len + horizon, :, :]
        x.append(x_i)
        y.append(y_i)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_graph_seq2seq_io_data_with_time2(df, batch_size, seq_len, horizon, num_nodes, scaler=None,
                                              add_time_in_day=True, add_day_in_week=False):
    """

    :param df:
    :param batch_size:
    :param seq_len:
    :param horizon:
    :param scaler:
    :param add_time_in_day:
    :param add_day_in_week:
    :return:
    x, y, both are 5-D tensors with size (epoch_size, batch_size, seq_len, num_sensors, input_dim).
    Adjacent batches are continuous sequence, i.e., x[i, j, :, :] is before x[i+1, j, :, :]
    """
    if scaler:
        df = scaler.transform(df)
    num_samples, _ = df.shape
    assert df.shape[1] == num_nodes
    data = df.values
    data = np.expand_dims(data, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    # data: (num_samples, num_nodes, num_features)
    data = np.concatenate(data_list, axis=-1)
    num_features = data.shape[-1]

    # Extract x and y
    epoch_size = num_samples - seq_len - horizon + 1
    x, y = [], []
    for i in range(epoch_size):
        x_i = data[i: i + seq_len, ...]
        y_i = data[i + seq_len: i + seq_len + horizon, ...]
        x.append(x_i)
        y.append(y_i)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    epoch_size //= batch_size
    x = x[:batch_size * epoch_size, ...]
    y = y[:batch_size * epoch_size, ...]
    x = x.reshape(epoch_size, batch_size, seq_len, num_nodes, num_features)
    y = y.reshape(epoch_size, batch_size, horizon, num_nodes, num_features)
    return x, y


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def round_down(num, divisor):
    return num - (num % divisor)


def separate_seasonal_trend_and_residual(df, period, test_ratio=0.2, null_val=0., epsilon=1e-4):
    """

    :param df:
    :param period:
    :param test_ratio: only use training part to calculate the average.
    :param null_val: indicator of missing values. Assuming null_val
    :param epsilon:
    :return:
    """
    n_sample, n_sensor = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    seasonal_trend = np.zeros((period, n_sensor), dtype=np.float32)
    for i in range(period):
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]
        seasonal_trend[i, :] = historical[historical != null_val].mean()
    n_repeat = (n_sample + period - 1) // period
    data = np.tile(seasonal_trend, [n_repeat, 1])[:n_sample, :]
    seasonal_df = pd.DataFrame(data, index=df.index, columns=df.columns)
    # Records where null value is happening.

    missing_ind = df == null_val
    residual_df = df - seasonal_df
    residual_df[residual_df == null_val] += epsilon
    residual_df[missing_ind] = null_val
    return seasonal_df, residual_df


def train_test_split(x, y, test_ratio=0.2, random=False, granularity=1):
    """
    This just splits data to training and testing parts. Default 80% train, 20% test
    Format : data is in compressed sparse row format

    Args:
        :param x data
        :param y label
        :param test_ratio:
        :param random: whether to randomize the input data.
        :param granularity:

    """
    perms = np.arange(0, x.shape[0])
    if random:
        perms = np.random.permutation(np.arange(0, x.shape[0]))
    n_train = round_down(int(round(x.shape[0] * (1 - test_ratio))), granularity)
    n_test = round_down(x.shape[0] - n_train, granularity)
    x_train, y_train = x.take(perms[:n_train], axis=0), y.take(perms[:n_train], axis=0)
    x_test, y_test = x.take(perms[n_train:n_train + n_test], axis=0), y.take(perms[n_train:n_train + n_test], axis=0)
    return (x_train, y_train), (x_test, y_test)


def train_val_test_split_df(df, val_ratio=0.1, test_ratio=0.2):
    n_sample, _ = df.shape
    n_val = int(round(n_sample * val_ratio))
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_val - n_test
    train_data, val_data, test_data = df.iloc[:n_train, :], df.iloc[n_train: n_train + n_val, :], df.iloc[-n_test:, :]
    return train_data, val_data, test_data
