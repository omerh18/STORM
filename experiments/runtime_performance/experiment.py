from src.feature_transformations import rocket_transform, rocket_based_blocks_transform, scale_features
from src.classifiers import get_rocket_classifier_by_name, get_storm_bilstm_classifier
from experiments.runtime_performance.config import RuntimeExperimentRunConfig
from src.representation_generation import generate_representation
from src.utils import read_specific_dataset, cross_validate
from tensorflow.keras.utils import to_categorical
from experiments import experiments_utils
import pandas as pd
import numpy as np
import time


def run_experiment(run_params: RuntimeExperimentRunConfig):
    np.random.seed(0)
    rep_times = []
    methods_times = []

    for dataset in run_params.datasets:
        print(f'NOW PROCESSING DATASET: {dataset}')

        print('READING DATA')

        labels_df, data_df, _ = read_specific_dataset(dataset)

        print('CONVERTING REP.')

        converted_mts_rep, series_labels, rep_time = generate_representation(labels_df, data_df)

        print(f'CONVERTED {dataset} IN {np.round(rep_time, 3)} sec')
        rep_time_run_summary = {
            'dataset': dataset,
            'rep_time': rep_time
        }
        rep_times.append(rep_time_run_summary)

        converted_mts_rep = np.array([s.T for s in converted_mts_rep])

        num_classes = labels_df['label'].nunique()

        del labels_df, data_df

        fold_idx = 0

        print(f'RUNNING {run_params.k}-FOLD CV')

        for X_train, y_train, X_test, y_test in cross_validate(converted_mts_rep, series_labels, k=run_params.k):

            print(f'***** FOLD: {fold_idx} *****')

            for use_multirocket in run_params.multirocket_usage:

                rocket_method = 'multirocket' if use_multirocket else 'minirocket'
                storm_method = f'storm {rocket_method}'

                for calc_first_order_diff in run_params.first_order_diff_usage:

                    if not use_multirocket:
                        if calc_first_order_diff != 0:
                            continue
                        X_train, X_test = np.float32(X_train), np.float32(X_test)
                    else:
                        X_train, X_test = np.float64(X_train), np.float64(X_test)

                    base_run_params = {
                        'dataset': dataset,
                        'fold': fold_idx,
                        'first_order_diff_used': calc_first_order_diff
                    }

                    for num_features in run_params.num_features_array:

                        print(f'METHOD: {rocket_method}, W/O FIRST ORDER DIFFERENCE SERIES: {calc_first_order_diff}, NUM FEATURES: {num_features}')
                        rocket_run_times_summary = {'method': rocket_method}
                        rocket_run_times_summary.update(run_rocket_on_fold(
                            X_train, X_test, y_train, use_multirocket,
                            calc_first_order_diff, num_features
                        ))
                        rocket_run_times_summary.update(base_run_params)
                        methods_times.append(rocket_run_times_summary)
                        print(f'RUN TIMES: {rocket_run_times_summary}')

                        for block_size in run_params.block_size_array:

                            print(f'METHOD: {storm_method}, W/O FIRST ORDER DIFFERENCE SERIES: {calc_first_order_diff}, NUM FEATURES: {num_features}, BLOCK SIZE: {block_size}')
                            storm_run_times_summary = {'method': storm_method}
                            storm_run_times_summary.update(run_storm_on_fold(
                                X_train, X_test, y_train, use_multirocket,
                                calc_first_order_diff, num_features, block_size,
                                run_params, num_classes
                            ))
                            storm_run_times_summary.update(base_run_params)
                            methods_times.append(storm_run_times_summary)
                            print(f'RUN TIMES: {storm_run_times_summary}')

            fold_idx += 1

            del X_train
            del y_train
            del X_test
            del y_test

        del converted_mts_rep

    rep_times_df, methods_times_df = finalize_experiment_results(rep_times, methods_times, run_params)
    return rep_times_df, methods_times_df


def run_storm_on_fold(
        X_train, X_test, y_train, use_multirocket, calc_first_order_diff, num_features, block_size, run_params, num_classes
):
    X_train_blocks_transformed, X_test_blocks_transformed, train_transform_time, test_transform_time, train_transform_block_time, test_transform_block_time = rocket_based_blocks_transform(
        use_multirocket, X_train, X_test, calc_first_order_diff, num_features, block_size
    )

    X_train_transform_scaled, X_test_transform_scaled = scale_features(
        X_train_blocks_transformed,
        X_test_blocks_transformed
    )

    del X_train_blocks_transformed
    del X_test_blocks_transformed

    y_train_softmax = to_categorical(y_train)

    best_model_filename = 'best_model.h5'
    model, callbacks = get_storm_bilstm_classifier(
        X_train_transform_scaled.shape[-1], num_classes, best_model_filename,
        print_summary=True, dense_size=run_params.dense_size, dropout=run_params.dropout,
        lstm_size=run_params.lstm_size, use_reg=run_params.use_reg, lstm_activation='tanh', bidirectional=True,
    )

    t = time.time()
    model.fit(
        X_train_transform_scaled,
        y_train_softmax,
        batch_size=None,
        epochs=run_params.num_epochs,
        verbose=False,
        callbacks=callbacks[2:]
    )
    train_cls_time = time.time() - t
    epoch_times = callbacks[2].times
    train_cls_epoch_time = np.mean(epoch_times[2:])

    t = time.time()
    model.predict(X_test_transform_scaled)
    test_cls_time = time.time() - t

    times = {
        'num_features': num_features,
        'block_size': block_size,
        'network_config': f'{run_params.dense_size},{run_params.dropout},{run_params.lstm_size},{run_params.use_reg}',
        'train_transform_time': train_transform_time,
        'test_transform_time': test_transform_time,
        'train_transform_block_time': train_transform_block_time,
        'test_transform_block_time': test_transform_block_time,
        'train_cls_time': train_cls_time,
        'train_cls_epoch_time': train_cls_epoch_time,
        'test_cls_time': test_cls_time
    }

    del X_train_transform_scaled
    del X_test_transform_scaled
    del model

    return times


def run_rocket_on_fold(
        X_train, X_test, y_train, use_multirocket, calc_first_order_diff, num_features
):
    X_train_transformed, X_test_transformed, train_transform_time, test_transform_time, _, _ = rocket_transform(
        use_multirocket, X_train, X_test, calc_first_order_diff, num_features
    )

    classifier = get_rocket_classifier_by_name('RIDGE')

    t = time.time()
    classifier.fit(X_train_transformed, y_train)
    train_cls_time = time.time() - t

    t = time.time()
    classifier.predict(X_test_transformed)
    test_cls_time = time.time() - t

    del X_train_transformed
    del X_test_transformed

    times = {
        'num_features': num_features,
        'block_size': np.nan,
        'network_config': np.nan,
        'train_transform_time': train_transform_time,
        'test_transform_time': test_transform_time,
        'train_transform_block_time': train_transform_time,
        'test_transform_block_time': test_transform_time,
        'train_cls_time': train_cls_time,
        'train_cls_epoch_time': np.nan,
        'test_cls_time': test_cls_time
    }

    return times


def finalize_experiment_results(rep_times, methods_times, run_params):
    rep_times_df = pd.DataFrame(rep_times)
    rep_times_df = rep_times_df[['dataset', 'rep_time']].sort_values(['dataset']).reset_index(drop=True)
    if run_params.write_csv:
        rep_times_df.to_csv(
            f'{experiments_utils.RESULTS_DIR}/results_times_rep_storm.csv',
            index=False
        )

    methods_times_df = pd.DataFrame(methods_times)
    methods_times_df = methods_times_df.groupby(
        ['dataset', 'method', 'first_order_diff_used', 'num_features', 'block_size', 'network_config'],
        dropna=False
    )[[
        'train_transform_time', 'test_transform_time', 'train_transform_block_time', 'test_transform_block_time',
        'train_cls_time', 'train_cls_epoch_time', 'test_cls_time'
    ]].mean().reset_index()
    methods_times_df = methods_times_df[[
        'dataset', 'method', 'first_order_diff_used', 'num_features', 'block_size', 'network_config',
        'train_transform_time', 'test_transform_time', 'train_transform_block_time', 'test_transform_block_time',
        'train_cls_time', 'train_cls_epoch_time', 'test_cls_time'
    ]].sort_values(
        ['dataset', 'method', 'first_order_diff_used', 'num_features', 'block_size', 'network_config']
    ).reset_index(drop=True)
    if run_params.write_csv:
        methods_times_df.to_csv(
            f'{experiments_utils.RESULTS_DIR}/results_times_methods.csv',
            index=False
        )

    return rep_times_df, methods_times_df
