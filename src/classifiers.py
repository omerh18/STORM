from keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, GlobalAveragePooling1D
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from keras import regularizers
from keras import Model
import tensorflow as tf
import numpy as np
import time
import keras


class TimingCallback(keras.callbacks.Callback):

    def __init__(self):
        self.times = None
        self.epoch_time_start = None
        super(keras.callbacks.Callback, self).__init__()

    def on_train_begin(self, _):
        self.times = []

    def on_epoch_begin(self, _, __):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, _, __):
        self.times.append(time.time() - self.epoch_time_start)


MODEL_FILE_PATH_TEMPLATE = '{}_{}_best_model.hdf5'


def get_storm_bilstm_classifier(shape, num_classes, best_model_filename, print_summary=True, dense_size=128, dropout=0,
                                lstm_size=64, pool=False, lstm_activation='tanh', use_dense=True, bidirectional=True,
                                use_reg=False):
    input_shape = (None, shape)
    input_layer = Input(input_shape)
    x = input_layer

    if use_dense:
        if use_reg:
            x = Dense(
                dense_size,
                activation='relu',
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            )(x)
        else:
            x = Dense(
                dense_size,
                activation='relu',
            )(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    if not pool:
        if bidirectional:
            x = Bidirectional(LSTM(lstm_size, activation=lstm_activation))(x)
        else:
            x = LSTM(lstm_size, activation=lstm_activation)(x)
    else:
        x = GlobalAveragePooling1D()(x)

    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=50,
        min_lr=0.00001
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=best_model_filename,
        monitor='val_accuracy',
        save_best_only=True
    )

    time_cb = TimingCallback()

    callbacks = [reduce_lr, model_checkpoint, time_cb]

    if print_summary:
        print(model.summary())

    return model, callbacks


def get_rocket_classifier_by_name(cls_type):
    if cls_type == 'RIDGE':
        cls = RidgeClassifierCV(
            # alphas=(0.1, 1, 10),
            alphas=np.logspace(-3, 3, 10),
            normalize=True,
            class_weight='balanced'
        )

    elif cls_type == 'LR':
        cls = LogisticRegression(
            solver='lbfgs', penalty='l2', random_state=0, C=1, max_iter=1000
        )  # , class_weight='balanced')

    else:
        raise Exception('Classifier not supported')

    return cls
