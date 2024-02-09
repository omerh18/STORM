from src.utils import cross_validate, read_specific_dataset, get_ridge_classifier_scores, get_keras_saved_model_classification_scores
from src.feature_transformations import rocket_transform, rocket_based_blocks_transform, scale_features
from experiments.classification_performance.config import ClassificationExperimentRunConfig
from src.classifiers import get_rocket_classifier_by_name, get_storm_bilstm_classifier
from src.representation_generation import generate_representation
from tensorflow.keras.utils import to_categorical
from experiments import experiments_utils
import pandas as pd
import numpy as np


def run_experiment(run_params: ClassificationExperimentRunConfig):

    np.random.seed(0)
    results = []

    for dataset in run_params.datasets:
        print(f'NOW PROCESSING DATASET: {dataset}')

        print('READING DATA')

        labels_df, data_df, _ = read_specific_dataset(dataset)

        print('CONVERTING REP.')

        converted_mts_rep, series_labels, _ = generate_representation(labels_df, data_df)

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

                    print(f'METHOD: {rocket_method}, W/O FIRST ORDER DIFFERENCE SERIES: {calc_first_order_diff}')
                    rocket_run_scores = {'method': rocket_method}
                    rocket_run_scores.update(run_rocket_on_fold(
                        X_train, X_test, y_train, y_test, use_multirocket,
                        calc_first_order_diff, run_params, num_classes
                    ))
                    rocket_run_scores.update(base_run_params)
                    results.append(rocket_run_scores)
                    print(f'SCORES: {rocket_run_scores}')

                    print(f'METHOD: {storm_method}, W/O FIRST ORDER DIFFERENCE SERIES: {calc_first_order_diff}')
                    storm_run_scores = {'method': storm_method}
                    storm_run_scores.update(run_storm_on_fold(
                        X_train, X_test, y_train, y_test, use_multirocket,
                        calc_first_order_diff, run_params, num_classes
                    ))
                    storm_run_scores.update(base_run_params)
                    results.append(storm_run_scores)
                    print(f'SCORES: {storm_run_scores}')

            fold_idx += 1

            del X_train
            del y_train
            del X_test
            del y_test

        del converted_mts_rep

    results_df = finalize_experiment_results(results, run_params)
    return results_df


def run_storm_on_fold(
        X_train, X_test, y_train, y_test, use_multirocket, calc_first_order_diff, run_params, num_classes
):
    X_train_blocks_transformed, X_test_blocks_transformed, _, _, _, _ = rocket_based_blocks_transform(
        use_multirocket, X_train, X_test, calc_first_order_diff, run_params.num_features, run_params.block_size
    )

    X_train_transform_scaled, X_test_transform_scaled = scale_features(
        X_train_blocks_transformed,
        X_test_blocks_transformed
    )

    del X_train_blocks_transformed
    del X_test_blocks_transformed

    y_train_softmax, y_test_softmax = to_categorical(y_train), to_categorical(y_test)

    best_model_filename = 'best_model.h5'
    model, callbacks = get_storm_bilstm_classifier(
        X_train_transform_scaled.shape[-1], num_classes, best_model_filename,
        print_summary=True, dense_size=run_params.dense_size, dropout=run_params.dropout,
        lstm_size=run_params.lstm_size, use_reg=run_params.use_reg, lstm_activation='tanh', bidirectional=True,
    )

    model.fit(
        X_train_transform_scaled,
        y_train_softmax,
        batch_size=None,
        epochs=run_params.num_epochs,
        verbose=False,
        validation_data=(X_test_transform_scaled, y_test_softmax),
        callbacks=callbacks
    )

    scores = {}

    for name, X, y in [
        ('train', X_train_transform_scaled, y_train),
        ('test', X_test_transform_scaled, y_test)
    ]:
        accuracy, auc = get_keras_saved_model_classification_scores(best_model_filename, X, y, num_classes)

        scores[f'{name}_accuracy'] = accuracy
        scores[f'{name}_auc'] = auc

    del X_train_transform_scaled
    del X_test_transform_scaled
    del model

    return scores


def run_rocket_on_fold(
        X_train, X_test, y_train, y_test, use_multirocket, calc_first_order_diff, run_params, num_classes
):
    X_train_transformed, X_test_transformed, _, _, _, _ = rocket_transform(
        use_multirocket, X_train, X_test, calc_first_order_diff, run_params.num_features
    )

    classifier = get_rocket_classifier_by_name('RIDGE')
    classifier.fit(X_train_transformed, y_train)

    scores = {}

    for name, X, y in [
        ('train', X_train_transformed, y_train),
        ('test', X_test_transformed, y_test)
    ]:
        accuracy, auc = get_ridge_classifier_scores(classifier, X, y, num_classes)

        scores[f'{name}_accuracy'] = accuracy
        scores[f'{name}_auc'] = auc

    del X_train_transformed
    del X_test_transformed

    return scores


def finalize_experiment_results(results, run_params):
    results_df = pd.DataFrame(results)

    results_df = results_df[[
        'dataset', 'fold', 'method', 'first_order_diff_used', 'train_accuracy', 'train_auc', 'test_accuracy', 'test_auc'
    ]].sort_values(
        ['dataset', 'fold', 'method', 'first_order_diff_used']
    ).reset_index(drop=True)

    if run_params.write_csv:
        dropout_str = str(run_params.dropout).replace('.', '')
        results_df.to_csv(
            f'{experiments_utils.RESULTS_DIR}/results_{run_params.dense_size}_{dropout_str}_{run_params.lstm_size}_{run_params.use_reg}_{run_params.num_features}_{run_params.block_size}_all.csv',
            index=False
        )

    return results_df
