from extended_multirocket import multirocket_multivariate
from extended_minirocket import minirocket_multivariate
import numpy as np
import sklearn
import time
import math


def scale_features(X_train_blocks_transformed, X_test_blocks_transformed, scale_method=sklearn.preprocessing.MinMaxScaler):
    X_train_blocks_transformed_scaled = np.zeros(X_train_blocks_transformed.shape, dtype=X_train_blocks_transformed.dtype)
    X_test_blocks_transformed_scaled = np.zeros(X_test_blocks_transformed.shape, dtype=X_test_blocks_transformed.dtype)

    for idx in range(X_train_blocks_transformed.shape[1]):
        scaler = scale_method()
        X_train_blocks_transformed_scaled[:, idx, :] = scaler.fit_transform(X_train_blocks_transformed[:, idx, :])
        X_test_blocks_transformed_scaled[:, idx, :] = scaler.transform(X_test_blocks_transformed[:, idx, :])

    return X_train_blocks_transformed_scaled, X_test_blocks_transformed_scaled


def rocket_based_blocks_transform(use_multirocket, X_train, X_test, calc_first_order_diff, num_features, block_size):
    train_times, test_times = [], []
    X_train_blocks_transformed, X_test_blocks_transformed = None, None
    num_blocks = math.ceil(X_train.shape[-1] / block_size)
    num_blocks = num_blocks if (num_blocks - 1) * block_size + 10 < X_train.shape[-1] else num_blocks - 1

    rocket_base_parameters, rocket_fo_diff_parameters = None, None
    for i in range(num_blocks):

        X_train_i = X_train[:, :, i * block_size:(i + 1) * block_size]
        X_test_i = X_test[:, :, i * block_size:(i + 1) * block_size]

        if not use_multirocket:

            if i == 0 or (i == num_blocks - 1 and X_train_i.shape[-1] != block_size):
                rocket_base_parameters = minirocket_multivariate.fit(
                    X_train_i, num_features=num_features, max_dilations_per_kernel=32, kernel_length=9,
                    # max_channels_per_kernel=2000, random_num_channels=False, channels_per_kernel=10
                )

            t = time.time()
            X_train_transform_i = minirocket_multivariate.transform(X_train_i, rocket_base_parameters)
            train_time = time.time() - t

            t = time.time()
            X_test_transform_i = minirocket_multivariate.transform(X_test_i, rocket_base_parameters)
            test_time = time.time() - t

        else:
            X_train_i_fo_diff = np.diff(X_train_i, 1)

            if i == 0 or (i == num_blocks - 1 and X_train_i.shape[-1] != block_size):
                rocket_base_parameters = multirocket_multivariate.fit(
                    X_train_i, num_features, kernel_length=9, max_dilations_per_kernel=32,
                    # max_channels_per_kernel=2000, random_num_channels=False, channels_per_kernel=10
                )
                rocket_fo_diff_parameters = multirocket_multivariate.fit(
                    X_train_i_fo_diff, num_features, kernel_length=9, max_dilations_per_kernel=32,
                    # max_channels_per_kernel=2000, random_num_channels=False, channels_per_kernel=10
                )

            t = time.time()
            X_train_transform_i = multirocket_multivariate.transform(
                X_train_i, X_train_i_fo_diff, rocket_base_parameters, rocket_fo_diff_parameters, 4,
                calc_first_order_diff=calc_first_order_diff
            )
            train_time = time.time() - t

            X_test_i_fo_diff = np.diff(X_test_i, 1)

            t = time.time()
            X_test_transform_i = multirocket_multivariate.transform(
                X_test_i, X_test_i_fo_diff, rocket_base_parameters, rocket_fo_diff_parameters, 4,
                calc_first_order_diff=calc_first_order_diff
            )
            test_time = time.time() - t

        if X_train_blocks_transformed is None:
            X_train_blocks_transformed = np.zeros(
                (X_train.shape[0], num_blocks, X_train_transform_i.shape[-1]), dtype=X_train_transform_i.dtype
            )
            X_test_blocks_transformed = np.zeros(
                (X_test.shape[0], num_blocks, X_test_transform_i.shape[-1]), dtype=X_test_transform_i.dtype
            )

        X_train_blocks_transformed[:, i, :] = X_train_transform_i
        X_test_blocks_transformed[:, i, :] = X_test_transform_i

        train_times.append(train_time)
        test_times.append(test_time)

    return X_train_blocks_transformed, X_test_blocks_transformed, \
        np.sum(train_times), np.sum(test_times), np.mean(train_times), np.mean(test_times)


def rocket_transform(use_multirocket, X_train, X_test, calc_first_order_diff, num_features):
    if not use_multirocket:
        rocket_base_parameters = minirocket_multivariate.fit(
            X_train, num_features=num_features, max_dilations_per_kernel=32, kernel_length=9,
            # max_channels_per_kernel=2000, #random_num_channels=False, #channels_per_kernel=10
        )
        t = time.time()
        X_train_transformed = minirocket_multivariate.transform(X_train, rocket_base_parameters)
        train_time = time.time() - t

        t = time.time()
        X_test_transformed = minirocket_multivariate.transform(X_test, rocket_base_parameters)
        test_time = time.time() - t

    else:
        X_train_i_fo_diff = np.diff(X_train, 1)
        rocket_base_parameters = multirocket_multivariate.fit(
            X_train, num_features, kernel_length=9, max_dilations_per_kernel=32,
            # max_channels_per_kernel=2000, random_num_channels=False, channels_per_kernel=10
        )
        rocket_fo_diff_parameters = multirocket_multivariate.fit(
            X_train_i_fo_diff, num_features, kernel_length=9, max_dilations_per_kernel=32,
            # max_channels_per_kernel=2000, random_num_channels=False, channels_per_kernel=10
        )
        t = time.time()
        X_train_transformed = multirocket_multivariate.transform(
            X_train, X_train_i_fo_diff, rocket_base_parameters, rocket_fo_diff_parameters, 4, calc_first_order_diff=calc_first_order_diff
        )
        train_time = time.time() - t

        X_test_i_fo_diff = np.diff(X_test, 1)

        t = time.time()
        X_test_transformed = multirocket_multivariate.transform(
            X_test, X_test_i_fo_diff, rocket_base_parameters, rocket_fo_diff_parameters, 4, calc_first_order_diff=calc_first_order_diff
        )
        test_time = time.time() - t

    return X_train_transformed, X_test_transformed, train_time, test_time, train_time, test_time
