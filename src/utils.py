from experiments.experiments_utils import SingleRunConfig
from sklearn.utils.extmath import softmax
from simple_parsing import ArgumentParser
import pandas as pd
import numpy as np
import sklearn
import keras
import os


DATASETS_DIR = 'data'
DATA_SEPARATOR = '\t'


def read_labels_file(dataset):
    labels_file = f'{DATASETS_DIR}/{dataset}/labels.txt'

    with open(labels_file, 'r') as f:

        labels = f.readlines()

        labels = [label.strip().split(DATA_SEPARATOR) for label in labels]

        labels = pd.DataFrame([
            {
                'entity': int(label[0]),
                'label': int(label[1])
            } for label in labels
        ])

    return labels


def read_data_file(dataset):
    data_file = f'{DATASETS_DIR}/{dataset}/data.txt'
    
    with open(data_file, 'r') as f:
        
        data = f.readlines()
        
        data = [sti.strip().split(DATA_SEPARATOR) for sti in data]
        
        data = pd.DataFrame([
            {
                'entity': int(sti[0]),
                'symbol': int(sti[1]),
                'start': int(sti[2]),
                'finish': int(sti[3])
            } for sti in data
        ])
    
    return data	


def read_specific_dataset(dataset):
    labels = read_labels_file(dataset)

    data = read_data_file(dataset)

    return labels, data, dataset


def read_all_datasets():
    for d in os.scandir(DATASETS_DIR):
        
        if d.is_dir() and not d.name.startswith('.'):

            dataset = d.name

            yield read_specific_dataset(dataset)


def cross_validate(X, y, k):
    labels_indices = [
        [sample_idx for sample_idx, idx_class_label in enumerate(y) if idx_class_label == class_label]
        for class_label in set(y)
    ]

    folds_indices = [
        [
            sample_idx
            for label_indices in labels_indices
            for sample_idx in label_indices[fold_idx * len(label_indices) // k: (fold_idx + 1) * len(label_indices) // k]
        ]
        for fold_idx in range(k)
    ]

    for fold_idx in range(k):
        non_fold_indices = set([i for i in range(len(y))]) - set(folds_indices[fold_idx])
        X_train = np.array([X[i] for i in non_fold_indices])
        X_test = np.array([X[i] for i in folds_indices[fold_idx]])
        y_train = np.array([y[i] for i in non_fold_indices])
        y_test = np.array([y[i] for i in folds_indices[fold_idx]])
        yield X_train, y_train, X_test, y_test


def get_classification_scores(y, y_pred, y_pred_probas, num_classes):
    accuracy = sklearn.metrics.accuracy_score(y, y_pred)

    if num_classes > 2:
        auc = sklearn.metrics.roc_auc_score(y, y_pred_probas, multi_class='ovo', average='macro')
    else:
        auc = sklearn.metrics.roc_auc_score(y, y_pred)

    return accuracy, auc


def get_keras_saved_model_classification_scores(model_file_path, X, y, num_classes):
    model = keras.models.load_model(model_file_path, compile=True)

    y_pred_probas = model.predict(X)

    y_pred = np.argmax(y_pred_probas, axis=-1)

    return get_classification_scores(y, y_pred, y_pred_probas, num_classes)


def get_ridge_classifier_scores(classifier, X, y, num_classes):
    accuracy = classifier.score(X, y)

    if num_classes > 2:
        auc = sklearn.metrics.roc_auc_score(y, softmax(classifier.decision_function(X)), multi_class='ovo', average='macro')
    else:
        auc = sklearn.metrics.roc_auc_score(y, classifier.predict(X))

    return accuracy, auc


def parse_args(args):

    parser = ArgumentParser(args)
    parser.add_arguments(SingleRunConfig, dest='run_params')
    args = parser.parse_args()

    return args.run_params
