from src.utils import read_specific_dataset, cross_validate, get_keras_saved_model_classification_scores, parse_args
from src.feature_transformations import rocket_based_blocks_transform, scale_features
from src.representation_generation import generate_representation
from experiments.experiments_utils import SingleRunConfig
from src.classifiers import get_storm_bilstm_classifier
from tensorflow.keras.utils import to_categorical
import numpy as np
import sys


NUM_EPOCHS = 200


if __name__ == '__main__':

    print('PARSING ARGUMENTS')

    run_params: SingleRunConfig = parse_args(sys.argv[1:])

    print(f'NOW PROCESSING DATASET: {run_params.dataset}')

    print('READING DATA')

    labels_df, stis_data_df, _ = read_specific_dataset(run_params.dataset)

    print('CONVERTING REP.')

    converted_mts_rep, series_labels, _ = generate_representation(labels_df, stis_data_df)

    converted_mts_rep = np.array([s.T for s in converted_mts_rep])

    num_classes = labels_df['label'].nunique()

    fold_idx = 0

    print(f'RUNNING {run_params.k}-FOLD CV')

    for X_train, y_train, X_test, y_test in cross_validate(converted_mts_rep, series_labels, k=run_params.k):

        print(f'***** FOLD: {fold_idx}, METHOD: {run_params.method}, W/O FIRST ORDER DIFFERENCE SERIES: {run_params.calc_first_order_diff} *****')

        if not run_params.use_multirocket:
            if run_params.calc_first_order_diff > 0:
                continue
            X_train, X_test = np.float32(X_train), np.float32(X_test)
        else:
            X_train, X_test = np.float64(X_train), np.float64(X_test)

        print('MAP: TRANSFORMING INTO FEATURE VECTORS')

        X_train_blocks_transformed, X_test_blocks_transformed, _, _, _, _ = rocket_based_blocks_transform(
            run_params.use_multirocket, X_train, X_test, run_params.calc_first_order_diff,
            run_params.num_features, run_params.block_size
        )

        print('SCALING FEATURE VECTORS')

        X_train_transform_scaled, X_test_transform_scaled = scale_features(
            X_train_blocks_transformed,
            X_test_blocks_transformed
        )

        print('DEFINING NETWORK ARCHITECTURE')

        y_train_softmax, y_test_softmax = to_categorical(y_train), to_categorical(y_test)

        best_model_filename = 'best_model.h5'
        model, callbacks = get_storm_bilstm_classifier(
            X_train_transform_scaled.shape[-1], num_classes, best_model_filename,
            print_summary=True, dense_size=run_params.dense_size, dropout=run_params.dropout,
            lstm_size=run_params.lstm_size, use_reg=run_params.use_reg, lstm_activation='tanh', bidirectional=True,
        )

        print('REDUCE: TRAINING NETWORK')

        model.fit(
            X_train_transform_scaled,
            y_train_softmax,
            batch_size=None,
            epochs=NUM_EPOCHS,
            verbose=False,
            validation_data=(X_test_transform_scaled, y_test_softmax),
            callbacks=callbacks
        )

        print('RESULTS:')

        for name, X, y in [('train', X_train_transform_scaled, y_train), ('test', X_test_transform_scaled, y_test)]:
            accuracy, auc = get_keras_saved_model_classification_scores(best_model_filename, X, y, num_classes)
            print(f'{name} accuracy: {accuracy}, {name} auc: {auc}')

        fold_idx += 1
