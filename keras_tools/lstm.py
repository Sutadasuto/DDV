#####
# To deal with random initializations
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
#####

from keras.layers import Input, Dense, LSTM, CuDNNLSTM, Activation, Dropout, TimeDistributed, Bidirectional
from keras.models import Model
from keras.models import Sequential
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras_tools.attention_layers import Attention, AttentionWithContext
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut

import copy
import csv
import keras
import keras.backend as K
import keras_tools.lmnn as lmnn
import keras_tools.sequences as sequences
import keras_tools.validation as metrics
import numpy as np
import os
import pandas


def create_basic_lstm(hu=20, timesteps=1, data_dim=1, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (batch_size, timesteps, data_dim)
    # create model
    model = Sequential()
    if gpu:
        model.add(CuDNNLSTM(hu, input_shape=(timesteps, data_dim)))
    else:
        model.add(LSTM(hu, input_shape=(timesteps, data_dim)))
    if dropout is float:
        model.add(Dropout(dropout))
    model.add(Dense(output, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_basic_lstm_double_dense(hu=20, timesteps=1, data_dim=1, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (batch_size, timesteps, data_dim)
    # create model
    model = Sequential()
    if gpu:
        model.add(CuDNNLSTM(hu, input_shape=(timesteps, data_dim)))
    else:
        model.add(LSTM(hu, input_shape=(timesteps, data_dim)))
    if dropout is float:
        model.add(Dropout(dropout))
    model.add(Dense(int(hu / 2)))
    model.add(Dense(output, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_attention_lstm(hu=20, timesteps=1, data_dim=1, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (batch_size, timesteps, data_dim)
    # create model
    model = Sequential()
    if gpu:
        model.add(CuDNNLSTM(hu, input_shape=(timesteps, data_dim), return_sequences=True))
    else:
        model.add(LSTM(hu, input_shape=(timesteps, data_dim), return_sequences=True))
    if dropout is float:
        model.add(Dropout(dropout))
    model.add(Attention())
    model.add(Dense(output, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_attention_context_lstm(hu=20, timesteps=1, data_dim=1, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (batch_size, timesteps, data_dim)
    # create model
    model = Sequential()
    if gpu:
        model.add(CuDNNLSTM(hu, input_shape=(timesteps, data_dim), return_sequences=True))
    else:
        model.add(LSTM(hu, input_shape=(timesteps, data_dim), return_sequences=True))
    if dropout is float:
        model.add(Dropout(dropout))
    model.add(AttentionWithContext())
    model.add(Dense(output, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_multidata_basic_lstm(input_shapes, hu=20, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (timesteps, data_dim)
    seq_shape = []
    for shape in input_shapes:
        seq_shape.append(shape)
    n = len(input_shapes)

    # Define n input_data sequences
    seq = []
    for i in range(n):
        seq_i = Input(seq_shape[i])
        seq.append(seq_i)
    cat = keras.layers.concatenate(seq, axis=-1)
    # Create model
    if gpu:
        lstm = CuDNNLSTM(hu, input_shape=(input_shapes[0][0], input_shapes[0][1] + input_shapes[1][1]))(cat)
    else:
        lstm = LSTM(hu, input_shape=(input_shapes[0][0], input_shapes[0][1] + input_shapes[1][1]))(cat)
    if dropout is float:
        lstm = Dropout(dropout)(lstm)
    dense = Dense(output, activation="sigmoid")(lstm)
    model = Model(seq, dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_multidata_basic_lstm_double_dense(input_shapes, hu=20, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (timesteps, data_dim)
    seq_shape = []
    for shape in input_shapes:
        seq_shape.append(shape)
    n = len(input_shapes)

    # Define n input_data sequences
    seq = []
    for i in range(n):
        seq_i = Input(seq_shape[i])
        seq.append(seq_i)
    cat = keras.layers.concatenate(seq, axis=-1)
    # Create model
    if gpu:
        lstm = CuDNNLSTM(hu, input_shape=(input_shapes[0][0], input_shapes[0][1] + input_shapes[1][1]))(cat)
    else:
        lstm = LSTM(hu, input_shape=(input_shapes[0][0], input_shapes[0][1] + input_shapes[1][1]))(cat)
    if dropout is float:
        lstm = Dropout(dropout)(lstm)
    dense_1 = Dense(int(hu / 2))(lstm)
    dense_2 = Dense(output, activation="sigmoid")(dense_1)
    model = Model(seq, dense_2)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_multidata_attention_lstm(input_shapes, hu=20, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (num_streams, timesteps, data_dim)
    seq_shape = []
    for shape in input_shapes:
        seq_shape.append(shape)
    n = len(input_shapes)

    # Define n input_data sequences
    seq = []
    for i in range(n):
        seq_i = Input(seq_shape[i])
        seq.append(seq_i)
    cat = keras.layers.concatenate(seq, axis=-1)
    # Create model
    if gpu:
        lstm = CuDNNLSTM(hu, input_shape=(input_shapes[0][0], input_shapes[0][1] + input_shapes[1][1]),
                         return_sequences=True)(cat)
    else:
        lstm = LSTM(hu, input_shape=(input_shapes[0][0], input_shapes[0][1] + input_shapes[1][1]),
                    return_sequences=True)(cat)
    if dropout is float:
        lstm = Dropout(dropout)(lstm)
    result, attention = Attention(return_attention=True)(lstm)
    dense = Dense(output, activation="sigmoid")(result)
    model = Model(seq, dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_multidata_attention_context_lstm(input_shapes, hu=20, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (num_streams, timesteps, data_dim)
    seq_shape = []
    for shape in input_shapes:
        seq_shape.append(shape)
    n = len(input_shapes)

    # Define n input_data sequences
    seq = []
    for i in range(n):
        seq_i = Input(seq_shape[i])
        seq.append(seq_i)
    cat = keras.layers.concatenate(seq, axis=-1)
    # Create model
    if gpu:
        lstm = CuDNNLSTM(hu, input_shape=(input_shapes[0][0], input_shapes[0][1] + input_shapes[1][1]),
                         return_sequences=True)(cat)
    else:
        lstm = LSTM(hu, input_shape=(input_shapes[0][0], input_shapes[0][1] + input_shapes[1][1]),
                    return_sequences=True)(cat)
    if dropout is float:
        lstm = Dropout(dropout)(lstm)
    result, attention = AttentionWithContext(return_attention=True)(lstm)
    dense = Dense(output, activation="sigmoid")(result)
    model = Model(seq, dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_multistream_basic_lstm(input_shapes, hu=20, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (num_streams, timesteps, data_dim)
    seq_shape = []
    for shape in input_shapes:
        seq_shape.append(shape)
    n = len(input_shapes)

    # Create a LSTM for each stream
    seq = []
    lstms = []
    for i in range(n):
        input_data = Input(seq_shape[i])
        if gpu:
            lstm = CuDNNLSTM(hu, input_shape=input_shapes[i])(input_data)
        else:
            lstm = LSTM(hu, input_shape=input_shapes[i])(input_data)
        if dropout is float:
            lstm = Dropout(dropout)(lstm)
        seq.append(input_data)
        lstms.append(lstm)
    # Concatenate independent streams
    cat = keras.layers.concatenate(lstms, axis=-1)
    dense = Dense(output, activation="sigmoid")(cat)
    model = Model(seq, dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_multistream_basic_lstm_double_dense(input_shapes, hu=20, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (num_streams, timesteps, data_dim)
    seq_shape = []
    for shape in input_shapes:
        seq_shape.append(shape)
    n = len(input_shapes)

    # Create a LSTM for each stream
    seq = []
    lstms = []
    for i in range(n):
        input_data = Input(seq_shape[i])
        if gpu:
            lstm = CuDNNLSTM(hu, input_shape=input_shapes[i])(input_data)
        else:
            lstm = LSTM(hu, input_shape=input_shapes[i])(input_data)
        if dropout is float:
            lstm = Dropout(dropout)(lstm)
        seq.append(input_data)
        lstms.append(lstm)
    # Concatenate independent streams
    cat = keras.layers.concatenate(lstms, axis=-1)
    dense_1 = Dense(int(hu / 2))(cat)
    dense_2 = Dense(output, activation="sigmoid")(dense_1)
    model = Model(seq, dense_2)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_multistream_attention_lstm(input_shapes, hu=20, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (num_streams, timesteps, data_dim)
    seq_shape = []
    for shape in input_shapes:
        seq_shape.append(shape)
    n = len(input_shapes)

    # Create a LSTM for each stream
    seq = []
    lstms = []
    for i in range(n):
        input_data = Input(seq_shape[i])
        if gpu:
            lstm = CuDNNLSTM(hu, input_shape=input_shapes[i], return_sequences=True)(input_data)
        else:
            lstm = LSTM(hu, input_shape=input_shapes[i], return_sequences=True)(input_data)
        if dropout is float:
            lstm = Dropout(dropout)(lstm)
        # Add attention in each stream
        result, attention = Attention(return_attention=True)(lstm)
        seq.append(input_data)
        lstms.append(result)
    # Concatenate independent streams
    cat = keras.layers.concatenate(lstms, axis=-1)
    dense = Dense(output, activation="sigmoid")(cat)
    model = Model(seq, dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_multistream_attention_context_lstm(input_shapes, hu=20, output=1, dropout=None, gpu=True):
    # expected input_data data shape: (num_streams, timesteps, data_dim)
    seq_shape = []
    for shape in input_shapes:
        seq_shape.append(shape)
    n = len(input_shapes)

    # Create a LSTM for each stream
    seq = []
    lstms = []
    for i in range(n):
        input_data = Input(seq_shape[i])
        if gpu:
            lstm = CuDNNLSTM(hu, input_shape=input_shapes[i], return_sequences=True)(input_data)
        else:
            lstm = LSTM(hu, input_shape=input_shapes[i], return_sequences=True)(input_data)
        if dropout is float:
            lstm = Dropout(dropout)(lstm)
        # Add attention in each stream
        result, attention = AttentionWithContext(return_attention=True)(lstm)
        seq.append(input_data)
        lstms.append(result)
    # Concatenate independent streams
    cat = keras.layers.concatenate(lstms, axis=-1)
    dense = Dense(output, activation="sigmoid")(cat)
    model = Model(seq, dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def architecture_1(input_shapes, hu=20, output=1, dropout=None, gpu=True):

    name = "Hierarchical Parallel Multistream LSTMs with context"
    mod_seq = []
    mod_representation = []
    for input_shape in input_shapes:
        seq_shape = []
        for shape in input_shape:
            seq_shape.append(shape)
        n = len(input_shape)

        # Create a LSTM for each view
        seq = []
        lstms = []
        for i in range(n):
            input_data = Input(seq_shape[i])
            if gpu:
                lstm = CuDNNLSTM(hu, input_shape=input_shape[i], return_sequences=True)(input_data)
            else:
                lstm = LSTM(hu, input_shape=input_shape[i], return_sequences=True)(input_data)
            if dropout is float:
                lstm = Dropout(dropout)(lstm)
            # Add attention in each stream
            result, attention = AttentionWithContext(return_attention=True)(lstm)
            seq.append(input_data)
            lstms.append(result)
        # Concatenate independent streams
        merge = keras.layers.add(lstms)
        mod_seq.append(seq)
        mod_representation.append(merge)
    video_representation = keras.layers.add(mod_representation)
    dense = Dense(output, activation="sigmoid")(video_representation)
    model = Model([item for sublist in mod_seq for item in sublist], dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, name


def architecture_1_2(input_shapes, hu=20, output=1, dropout=None, gpu=True):

    name = "Hierarchical Parallel Multistream Hadamard LSTMs with context"
    mod_seq = []
    mod_representation = []
    for input_shape in input_shapes:
        seq_shape = []
        for shape in input_shape:
            seq_shape.append(shape)
        n = len(input_shape)

        # Create a LSTM for each view
        seq = []
        lstms = []
        for i in range(n):
            input_data = Input(seq_shape[i])
            if gpu:
                lstm = CuDNNLSTM(hu, input_shape=input_shape[i], return_sequences=True)(input_data)
            else:
                lstm = LSTM(hu, input_shape=input_shape[i], return_sequences=True)(input_data)
            if dropout is float:
                lstm = Dropout(dropout)(lstm)
            # Add attention in each stream
            result, attention = AttentionWithContext(return_attention=True)(lstm)
            seq.append(input_data)
            lstms.append(result)
        # Concatenate independent streams
        merge = keras.layers.multiply(lstms)
        mod_seq.append(seq)
        mod_representation.append(merge)
    video_representation = keras.layers.multiply(mod_representation)
    dense = Dense(output, activation="sigmoid")(video_representation)
    model = Model([item for sublist in mod_seq for item in sublist], dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, name


def architecture_1_3(input_shapes, hu=20, output=1, dropout=None, gpu=True):

    name = "Hierarchical Parallel Multistream Attention LSTMs with context"
    mod_seq = []
    mod_representation = []
    for input_shape in input_shapes:
        seq_shape = []
        for shape in input_shape:
            seq_shape.append(shape)
        n = len(input_shape)

        # Create a LSTM for each view
        seq = []
        lstms = []
        for i in range(n):
            input_data = Input(seq_shape[i])
            if gpu:
                lstm = CuDNNLSTM(hu, input_shape=input_shape[i], return_sequences=True)(input_data)
            else:
                lstm = LSTM(hu, input_shape=input_shape[i], return_sequences=True)(input_data)
            if dropout is float:
                lstm = Dropout(dropout)(lstm)
            # Add attention in each stream
            result, attention = AttentionWithContext(return_attention=True)(lstm)
            seq.append(input_data)
            lstms.append(result)
        # Concatenate independent streams
        merge = keras.layers.concatenate(lstms)
        merge = keras.layers.Reshape((len(lstms), hu))(merge)
        views_attention, attention2 = AttentionWithContext(return_attention=True)(merge)
        mod_seq.append(seq)
        mod_representation.append(views_attention)
    mod_merge = keras.layers.concatenate(mod_representation)
    mod_merge = keras.layers.Reshape((len(mod_representation), hu))(mod_merge)
    video_representation, attention3 = AttentionWithContext(return_attention=True)(mod_merge)
    dense = Dense(output, activation="sigmoid")(video_representation)
    model = Model([item for sublist in mod_seq for item in sublist], dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, name


def architecture_2(input_shapes, hu=20, output=1, dropout=None, gpu=True):

    name = "Multiview LSTMs with context"
    seq_shape = []
    n = 0
    for input_shape in input_shapes:
        for shape in input_shape:
            seq_shape.append(shape)
        n += len(input_shape)

    # Create a LSTM for each view
    seq = []
    views = []
    for i in range(n):
        input_data = Input(seq_shape[i])
        if gpu:
            lstm = CuDNNLSTM(hu, input_shape=seq_shape[i], return_sequences=True)(input_data)
        else:
            lstm = LSTM(hu, input_shape=seq_shape[i], return_sequences=True)(input_data)
        if dropout is float:
            lstm = Dropout(dropout)(lstm)
        # Add attention in each stream
        result, attention = AttentionWithContext(return_attention=True)(lstm)
        view_representation = Dense(output, activation="tanh")(result)
        seq.append(input_data)
        views.append(view_representation)
    video_representation = keras.layers.concatenate(views)
    dense_out = Dense(output, activation="sigmoid")(video_representation)
    model = Model(seq, dense_out)
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, name


def architecture_3(input_shapes, hu=20, output=1, dropout=None, gpu=True):

    name = "Multiview LSTMs with attention"
    seq_shape = []
    n = 0
    for input_shape in input_shapes:
        for shape in input_shape:
            seq_shape.append(shape)
        n += len(input_shape)

    # Create a LSTM for each view
    seq = []
    views = []
    for i in range(n):
        input_data = Input(seq_shape[i])
        if gpu:
            lstm = CuDNNLSTM(hu, input_shape=seq_shape[i], return_sequences=True)(input_data)
        else:
            lstm = LSTM(hu, input_shape=seq_shape[i], return_sequences=True)(input_data)
        if dropout is float:
            lstm = Dropout(dropout)(lstm)
        # Add attention in each stream
        result, attention = Attention(return_attention=True)(lstm)
        view_representation = Dense(output, activation="tanh")(result)
        seq.append(input_data)
        views.append(view_representation)
    video_representation = keras.layers.concatenate(views)
    dense_out = Dense(output, activation="sigmoid")(video_representation)
    model = Model(seq, dense_out)
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, name


def architecture_4(input_shapes, hu=20, output=1, dropout=None, gpu=True):

    name = "Multiview LSTMs with context binary loss"
    seq_shape = []
    n = 0
    for input_shape in input_shapes:
        for shape in input_shape:
            seq_shape.append(shape)
        n += len(input_shape)

    # Create a LSTM for each view
    seq = []
    views = []
    for i in range(n):
        input_data = Input(seq_shape[i])
        if gpu:
            lstm = CuDNNLSTM(hu, input_shape=seq_shape[i], return_sequences=True)(input_data)
        else:
            lstm = LSTM(hu, input_shape=seq_shape[i], return_sequences=True)(input_data)
        if dropout is float:
            lstm = Dropout(dropout)(lstm)
        # Add attention in each stream
        result, attention = AttentionWithContext(return_attention=True)(lstm)
        view_representation = Dense(output, activation="tanh")(result)
        seq.append(input_data)
        views.append(view_representation)
    video_representation = keras.layers.concatenate(views)
    dense_out = Dense(output, activation="sigmoid")(video_representation)
    model = Model(seq, dense_out)
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, name


def architecture_5(input_shapes, hu=20, output=1, dropout=None, gpu=True):

    name = "Multiview LSTMs with attention binary loss"
    seq_shape = []
    n = 0
    for input_shape in input_shapes:
        for shape in input_shape:
            seq_shape.append(shape)
        n += len(input_shape)

    # Create a LSTM for each view
    seq = []
    views = []
    for i in range(n):
        input_data = Input(seq_shape[i])
        if gpu:
            lstm = CuDNNLSTM(hu, input_shape=seq_shape[i], return_sequences=True)(input_data)
        else:
            lstm = LSTM(hu, input_shape=seq_shape[i], return_sequences=True)(input_data)
        if dropout is float:
            lstm = Dropout(dropout)(lstm)
        # Add attention in each stream
        result, attention = Attention(return_attention=True)(lstm)
        view_representation = Dense(output, activation="tanh")(result)
        seq.append(input_data)
        views.append(view_representation)
    video_representation = keras.layers.concatenate(views)
    dense_out = Dense(output, activation="sigmoid")(video_representation)
    model = Model(seq, dense_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, name


def architecture_6(input_shapes, hu=20, output=1, dropout=None, gpu=True):

    name = "Multiview Hierarchical LSTMs with context"
    mod_seq = []
    mod_representation = []
    for input_shape in input_shapes:
        seq_shape = []
        for shape in input_shape:
            seq_shape.append(shape)
        n = len(input_shape)

        # Create a LSTM for each view
        seq = []
        views = []
        for i in range(n):
            input_data = Input(seq_shape[i])
            if gpu:
                lstm = CuDNNLSTM(hu, input_shape=input_shape[i], return_sequences=True)(input_data)
            else:
                lstm = LSTM(hu, input_shape=input_shape[i], return_sequences=True)(input_data)
            if dropout is float:
                lstm = Dropout(dropout)(lstm)
            # Add attention in each stream
            result, attention = AttentionWithContext(return_attention=True)(lstm)
            view_rep = Dense(output, activation="tanh")(result)
            seq.append(input_data)
            views.append(view_rep)
        # Concatenate independent streams
        merge = keras.layers.concatenate(views)
        mod = Dense(output, activation="tanh")(merge)
        mod_seq.append(seq)
        mod_representation.append(mod)
    video_representation = keras.layers.concatenate(mod_representation)
    dense = Dense(output, activation="sigmoid")(video_representation)
    model = Model([item for sublist in mod_seq for item in sublist], dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, name


def test(gpu=False):
    seq_length = 5
    X = [[i + j for j in range(seq_length)] for i in range(100)]
    X_simple = [[i for i in range(4, 104)]]
    X = np.array(X)
    X_simple = np.array(X_simple)

    y = [[i + (i - 1) * .5 + (i - 2) * .2 + (i - 3) * .1 for i in range(4, 104)]]
    y = np.array(y)
    X_simple = X_simple.reshape((100, 1))
    X = X.reshape((100, 5, 1))
    y = y.reshape((100, 1))

    model = Sequential()
    if gpu:
        model.add(CuDNNLSTM(8, input_shape=(5, 1), return_sequences=False))
    else:
        model.add(LSTM(8, input_shape=(5, 1), return_sequences=False))
    model.add(Dense(2, activation="linear"))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    model.fit(X, y, epochs=2000, batch_size=5, validation_split=0.05, verbose=1);
    scores = model.evaluate(X, y, verbose=1, batch_size=5)
    print("Accurracy: {}".format(scores[1]))
    import matplotlib.pyplot as plt
    predict = model.predict(X)
    plt.plot(y, predict - y, 'C2')
    plt.ylim(ymax=3, ymin=-3)
    plt.show()


def single_modality(input_data, cv=10):
    # Input can be either a folder containing csv files or a .npy file
    if input_data.endswith(".npy"):
        loaded_array = np.load(input_data)
        X = loaded_array[0]
        Y = loaded_array[1]
    else:
        X, Y = sequences.get_input_sequences(input_data)

    seed = 8

    hu = 200
    dropout = 0.5
    epochs = 100
    batch_size = 32
    gpu = True
    classifier_basic = KerasClassifier(build_fn=create_basic_lstm, hu=hu, timesteps=X.shape[1], data_dim=X.shape[2],
                                       output=1,
                                       dropout=dropout, gpu=gpu, epochs=epochs, batch_size=batch_size, verbose=1)
    classifier_double_dense = KerasClassifier(build_fn=create_basic_lstm, hu=hu, timesteps=X.shape[1],
                                              data_dim=X.shape[2],
                                              output=1,
                                              dropout=dropout, gpu=gpu, epochs=epochs, batch_size=batch_size, verbose=1)
    classifier_attention = KerasClassifier(build_fn=create_attention_lstm, hu=hu, timesteps=X.shape[1],
                                           data_dim=X.shape[2],
                                           output=1,
                                           dropout=dropout, gpu=gpu, epochs=epochs, batch_size=batch_size, verbose=1)
    classifier_attention_context = KerasClassifier(build_fn=create_attention_context_lstm, hu=hu, timesteps=X.shape[1],
                                                   data_dim=X.shape[2],
                                                   output=1,
                                                   dropout=dropout, gpu=gpu, epochs=epochs, batch_size=batch_size,
                                                   verbose=1)
    if cv is int:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        folds = cv
    results_basic = cross_val_score(classifier_basic, X, Y, cv=folds, verbose=1)
    results_double_dense = cross_val_score(classifier_double_dense, X, Y, cv=folds, verbose=1)  # , n_jobs=-1)
    results_attention = cross_val_score(classifier_attention, X, Y, cv=folds, verbose=1)  # , n_jobs=-1)
    results_attention_context = cross_val_score(classifier_attention_context, X, Y, cv=folds, verbose=1)  # , n_jobs=-1)
    print("Database: %s" % os.path.split(input_data)[-2])
    print("Data: %s" % os.path.split(input_data)[-1])
    print("Hidden units: %s, Epochs: %s, Batch Size: %s, Dropout: %s" % (hu, epochs, batch_size, dropout))
    print("Result basic: %.2f%% (%.2f%%)" % (results_basic.mean() * 100, results_basic.std() * 100))
    print("Result double dense: %.2f%% (%.2f%%)" % (results_double_dense.mean() * 100, results_basic.std() * 100))
    print("Result attention: %.2f%% (%.2f%%)" % (results_attention.mean() * 100, results_attention.std() * 100))
    print("Result attention with context: %.2f%% (%.2f%%)" % (
        results_attention_context.mean() * 100, results_attention_context.std() * 100))

    with open(os.path.join(os.path.split(input_data)[-2], "keras_results_%s.txt" % os.path.split(input_data)[-1]),
              "w+") as output:
        output.write("Database: %s\n" % os.path.split(input_data)[-2])
        output.write("Data: %s\n" % os.path.split(input_data)[-1])
        output.write("Hidden units: %s, Epochs: %s, Batch Size: %s, Dropout: %s\n" % (hu, epochs, batch_size, dropout))
        output.write("Result basic: %.2f%% (%.2f%%)\n" % (results_basic.mean() * 100, results_basic.std() * 100))
        output.write(
            "Result double dense: %.2f%% (%.2f%%)\n" % (results_double_dense.mean() * 100, results_basic.std() * 100))
        output.write(
            "Result attention: %.2f%% (%.2f%%)\n" % (results_attention.mean() * 100, results_attention.std() * 100))
        output.write("Result attention with context: %.2f%% (%.2f%%)\n" % (
            results_attention_context.mean() * 100, results_attention_context.std() * 100))


def modalities(inputs, cv=10, seq_reduction="padding", reduction="avg", output_folder=None, hu=50, dropout=None,
                  epochs=100, batch_size=16, gpu=True, plot="", scoring="roc_auc", feat_standardization=False, verbose=2):
    # Input can be either a folder containing csv files or a .npy file
    if inputs is str:
        if inputs.endswith(".npy"):
            loaded_array = np.load(inputs)
            X = loaded_array[0]
            Y = loaded_array[1]
    else:
        if seq_reduction == "padding":
            length = reduction
        else:
            length = None
        X = []
        Y = []
        for stream_idx in range(len(inputs)):
            X_idx, Y_idx = sequences.get_input_sequences(inputs[stream_idx], length)
            X.append(X_idx)
            Y.append(Y_idx)
        if seq_reduction == "padding":
            X = sequences.multiple_sequence_padding(X)
        elif seq_reduction == "kmeans":
            X = sequences.kmeans_seq_reduction(X, k=reduction)
        elif seq_reduction == "pad_means":
            X = sequences.multiple_sequence_padding_means(X, reduction)
        elif seq_reduction == "sync_kmeans":
            X = sequences.synchronize_views(X)
            X = sequences.kmeans_sync_seq_reduction(X, k=reduction)

    if output_folder is None:
        output_folder = os.path.split(inputs[0])[0]

    input_shapes = [x.shape[1:] for x in X]
    if type(cv) is int:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10)
        folds = [fold for fold in folds.split(X[0], Y[0])]
    elif cv == "loo":
        folds = [fold for fold in LeaveOneOut().split(X[0])]
    else:
        folds = cv

    model_builders = [create_basic_lstm, create_basic_lstm_double_dense, create_attention_lstm,
                      create_attention_context_lstm]
    labels = ["Basic", "Double dense", "Attention", "Attention with context"]
    with open(os.path.join(output_folder, "lstm_results_modalities_%s_%s.txt" % (seq_reduction, reduction)),
              "w+") as output_file:
        for stream_idx in range(len(inputs)):
            header = "Database: %s\nData: %s\nHidden units: %s, Epochs: %s, Batch Size: %s, Dropout: %s, Seq. reduction: %s, %s\n" % (
                os.path.split(inputs[stream_idx])[0], os.path.split(inputs[stream_idx])[1], hu, epochs, batch_size,
                dropout, seq_reduction, reduction)
            output_file.write(header)
            for idx, builder in enumerate(model_builders):
                classifier = KerasClassifier(build_fn=builder, hu=hu, timesteps=X[stream_idx].shape[1],
                                               data_dim=X[stream_idx].shape[2], output=1, dropout=dropout, gpu=gpu,
                                               epochs=epochs, batch_size=batch_size, verbose=2)
                result = cross_val_score(classifier, X[stream_idx], Y[stream_idx], scoring=scoring, cv=folds,
                                          verbose=1)
                if K.backend() == 'tensorflow':
                    K.clear_session()
                    del classifier
                print(header.strip())
                metrics.write_result(result, labels[idx], output_file)
            output_file.write("\n")

        # Multidata
        model_builders = [create_multidata_basic_lstm, create_multidata_basic_lstm_double_dense,
                          create_multidata_attention_lstm, create_multidata_attention_context_lstm]
        labels = ["Early fusion Basic", "Early fusion Double dense", "Early fusion Attention",
                  "Early fusion Attention with context"]

        streams = [os.path.split(i)[1] for i in inputs]
        header = "Database: %s\nData: %s\nHidden units: %s, Epochs: %s, Batch Size: %s, Dropout: %s, Seq. reduction: %s, %s\n" % (
            os.path.split(inputs[0])[0], " + ".join(streams), hu, epochs, batch_size,
            dropout, seq_reduction, reduction)
        print(header.strip())
        output_file.write(header)

        for idx, classifier in enumerate(model_builders):
            results, name = metrics.cross_val_score(classifier, X, Y, feat_standardization, folds, scoring, plot,
                                                    batch_size, epochs, verbose,
                                                    hu=hu, input_shapes=input_shapes, output=1, dropout=dropout, gpu=gpu)
            print(header.strip())
            metrics.write_result(results, labels[idx], output_file)
        output_file.write("\n")

        # Multistream
        model_builders = [create_multistream_basic_lstm, create_multistream_basic_lstm_double_dense,
                          create_multistream_attention_lstm, create_multistream_attention_context_lstm]
        labels = ["Late fusion Basic", "Late fusion Double dense", "Late fusion Attention",
                  "Late fusion Attention with context"]

        header = "Database: %s\nData: %s\nHidden units: %s, Epochs: %s, Batch Size: %s, Dropout: %s, Seq. reduction: %s, %s\n" % (
            os.path.split(inputs[0])[0], " + ".join(streams), hu, epochs, batch_size,
            dropout, seq_reduction, reduction)
        print(header.strip())
        output_file.write(header)

        for idx, classifier in enumerate(model_builders):
            results, name = metrics.cross_val_score(classifier, X, Y, feat_standardization, folds, scoring, plot,
                                                    batch_size, epochs, verbose,
                                                    hu=hu, input_shapes=input_shapes, output=1, dropout=dropout, gpu=gpu)
            print(header.strip())
            metrics.write_result(results, labels[idx], output_file)


def my_method(modalities, cv=10, seq_reduction="padding", reduction="avg", output_folder=None, hu=50, dropout=None,
              epochs=100, batch_size=16, gpu=True, plot=True, scoring="roc_auc", feat_standardization=False, verbose=1):
    if modalities is str:
        if modalities.endswith(".npy"):
            loaded_array = np.load(modalities)
            X = loaded_array[0]
            Y = loaded_array[1]
    else:
        if seq_reduction == "padding":
            length = reduction
        else:
            length = None
        X = []
        Y = []
        for stream_idx, stream in enumerate(modalities):
            X_m = []
            Y_m = []
            for view in stream:
                X_idx, Y_idx = sequences.get_input_sequences(view, length)
                X_m.append(X_idx)
                Y_m.append(Y_idx)
            X.append(X_m)
            Y.append(Y_m)
        if seq_reduction == "padding":
            for mod_idx, modality in enumerate(X):
                X[mod_idx] = sequences.multiple_sequence_padding(modality)
        elif seq_reduction == "kmeans":
            for mod_idx, modality in enumerate(X):
                X[mod_idx] = sequences.kmeans_seq_reduction(modality, k=reduction)
        elif seq_reduction == "pad_means":
            for mod_idx, modality in enumerate(X):
                X[mod_idx] = sequences.multiple_sequence_padding_means(modality, reduction)
        elif seq_reduction == "sync_kmeans":
            for mod_idx, modality in enumerate(X):
                modality = sequences.synchronize_views(modality)
                X[mod_idx] = sequences.kmeans_sync_seq_reduction(modality, k=reduction)

    if output_folder is None:
        output_folder = os.path.split(os.path.split(modalities[0][0])[0])[0]

    input_shapes = [[x.shape[1:] for x in X_m] for X_m in X]
    X = [item for sublist in X for item in sublist]
    Y = [item for sublist in Y for item in sublist]
    if type(cv) is int:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10)
        folds = [fold for fold in folds.split(X[0], Y[0])]
    elif cv == "loo":
        folds = [fold for fold in LeaveOneOut().split(X[0])]
    else:
        folds = cv

    architectures = [architecture_1_3]
    streams = [", ".join([os.path.split(i)[1] for i in modality]) for modality in modalities]
    if plot == True:
        plot = output_folder
    else:
        plot = ""

    with open(os.path.join(output_folder, "%s_%s_%s.txt" % ("fusions", seq_reduction, reduction)), "w+") as output_file:
        header = "Database: %s\nData: %s\nHidden units: %s, Epochs: %s, Batch Size: %s, Dropout: %s, Seq. reduction: %s, %s\n" % (
            os.path.split(os.path.split(modalities[0][0])[0])[0], " + ".join(streams), hu, epochs, batch_size,
            dropout, seq_reduction, reduction)
        print(header.strip())
        output_file.write(header)
        results = [["", header], ["", scoring]]
        for architecture in architectures:
            result, name = metrics.cross_val_score(lmnn.create_multistream_model, X, Y, feat_standardization, folds,
                                                   scoring, plot,
                                                   batch_size, epochs, verbose,
                                                   input_shapes=input_shapes, hu=hu, output=1, dropout=dropout, gpu=gpu,
                                                   single_model_generator=architecture_1)
            print(header.strip())
            metrics.write_result(result, name, output_file)
            results.append([name, result])
    return results


def views_encoding(modalities, cv=10, seq_reduction="padding", reduction="avg", output_folder=None,
              epochs=100, batch_size=16, gpu=True, plot="", scoring="roc_auc", feat_standardization=False, verbose=2):

    if modalities is str:
        if modalities.endswith(".npy"):
            loaded_array = np.load(modalities)
            X = loaded_array[0]
            Y = loaded_array[1]
    else:
        if seq_reduction == "padding":
            length = reduction
        else:
            length = None
        X = []
        Y = []
        view_names = []
        for stream_idx, stream in enumerate(modalities):
            X_m = []
            Y_m = []
            for view in stream:
                view_names.append(os.path.split(view)[-1].replace(".arff",""))
                X_idx, Y_idx = sequences.get_input_sequences(view, length)
                X_m.append(X_idx)
                Y_m.append(Y_idx)
            X.append(X_m)
            Y.append(Y_m)
        if seq_reduction == "padding":
            for mod_idx, modality in enumerate(X):
                X[mod_idx] = sequences.multiple_sequence_padding(modality)
        elif seq_reduction == "kmeans":
            for mod_idx, modality in enumerate(X):
                X[mod_idx] = sequences.kmeans_seq_reduction(modality, k=reduction)
        elif seq_reduction == "pad_means":
            for mod_idx, modality in enumerate(X):
                X[mod_idx] = sequences.multiple_sequence_padding_means(modality, reduction)
        elif seq_reduction == "sync_kmeans":
            for mod_idx, modality in enumerate(X):
                modality = sequences.synchronize_views(modality)
                X[mod_idx] = sequences.kmeans_sync_seq_reduction(modality, k=reduction)

    if output_folder is None:
        output_folder = os.path.split(os.path.split(modalities[0][0])[0])[0]

    input_shapes = [x.shape[1:] for X_m in X for x in X_m]
    X = [item for sublist in X for item in sublist]
    Y = [item for sublist in Y for item in sublist]
    if type(cv) is int:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10)
        folds = [fold for fold in folds.split(X[0], Y[0])]
    elif cv == "loo":
        folds = [fold for fold in LeaveOneOut().split(X[0])]
    else:
        folds = cv

    def create_encoding_lstm(hu, timesteps, data_dim, output, gpu=True):
        # expected input_data data shape: (batch_size, timesteps, data_dim)
        # create model
        model = Sequential()
        if gpu:
            model.add(CuDNNLSTM(hu, return_sequences=False, input_shape=(timesteps, data_dim)))
        else:
            model.add(LSTM(hu, return_sequences=False, input_shape=(timesteps, data_dim)))
        # model.add(Attention())
        model.add(Dense(output, activation="sigmoid"))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    with open(os.path.join(output_folder, "%s_%s_%s.csv" % ("view_encodings", seq_reduction, reduction)), "w+") as output_file:
        header = "Epochs: %s, Batch Size: %s, Seq. reduction: %s, %s\n" % (
            epochs, batch_size, seq_reduction, reduction)
        print(header.strip())
        output_file.write(header)
        result_matrix = [["", header], ["", scoring]]
        for view_idx, view in enumerate(X):
            target_labels = Y[view_idx]
            if len(target_labels.shape) == 1:
                output=1
            else:
                output = target_labels.shape[-1]
            results, name = metrics.cross_val_score(create_encoding_lstm, view, target_labels, feat_standardization, folds,
                                                   scoring, plot,
                                                   batch_size, epochs, verbose,
                                                   hu=50, timesteps=input_shapes[view_idx][-2],
                                                   data_dim=input_shapes[view_idx][-1], output=output, gpu=gpu)
            print("%s\n%s: %s" % (header, view_names[view_idx], round(results.mean(), 4)))
            result_matrix.append([view_names[view_idx], round(results.mean(), 4)])
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(result_matrix)


def views_grid_search(modalities, cv=10, seq_reduction="padding", reduction="avg", output_folder=None,
              gpu=True, scoring="roc_auc", verbose=2):

    if modalities is str:
        if modalities.endswith(".npy"):
            loaded_array = np.load(modalities)
            X = loaded_array[0]
            Y = loaded_array[1]
    else:
        if seq_reduction == "padding":
            length = reduction
        else:
            length = None
        X = []
        Y = []
        view_names = []
        for stream_idx, stream in enumerate(modalities):
            X_m = []
            Y_m = []
            for view in stream:
                view_names.append(os.path.split(view)[-1].replace(".arff",""))
                X_idx, Y_idx = sequences.get_input_sequences(view, length)
                X_m.append(X_idx)
                Y_m.append(Y_idx)
            X.append(X_m)
            Y.append(Y_m)
        if seq_reduction == "padding":
            for mod_idx, modality in enumerate(X):
                X[mod_idx] = sequences.multiple_sequence_padding(modality)
        elif seq_reduction == "kmeans":
            for mod_idx, modality in enumerate(X):
                X[mod_idx] = sequences.kmeans_seq_reduction(modality, k=reduction)
        elif seq_reduction == "pad_means":
            for mod_idx, modality in enumerate(X):
                X[mod_idx] = sequences.multiple_sequence_padding_means(modality, reduction)
        elif seq_reduction == "sync_kmeans":
            for mod_idx, modality in enumerate(X):
                modality = sequences.synchronize_views(modality)
                X[mod_idx] = sequences.kmeans_sync_seq_reduction(modality, k=reduction)

    if output_folder is None:
        output_folder = os.path.split(os.path.split(modalities[0][0])[0])[0]

    input_shapes = [x.shape[1:] for X_m in X for x in X_m]
    X = [item for sublist in X for item in sublist]
    Y = [item for sublist in Y for item in sublist]
    if type(cv) is int:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10)
        folds = [fold for fold in folds.split(X[0], Y[0])]
    elif cv == "loo":
        folds = [fold for fold in LeaveOneOut().split(X[0])]
    else:
        folds = cv

    def create_encoding_lstm(hu, time_steps, data_dim, output, dropout=0, gpu=True):
        # create model
        model = Sequential()
        if gpu:
            model.add(CuDNNLSTM(hu, return_sequences=False, input_shape=(time_steps, data_dim)))
        else:
            model.add(LSTM(hu, return_sequences=False, input_shape=(time_steps, data_dim)))
        # model.add(Attention())
        model.add(Dropout(dropout, seed=0))
        model.add(Dense(output, activation="sigmoid"))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    from sklearn.model_selection import GridSearchCV
    with open(os.path.join(output_folder, "grid_search_results.txt"), "w+") as output_file:
        for view_idx, view in enumerate(X):
            target_labels = Y[view_idx]
            if len(target_labels.shape) == 1:
                output = 1
            else:
                output = target_labels.shape[-1]
            data_shape = view.shape
            data_dim = data_shape[-1]
            time_steps = data_shape[-2]
            model = KerasClassifier(create_encoding_lstm, gpu=gpu, verbose=verbose,
                                    output=output, time_steps=time_steps, data_dim=data_dim)

            epochs = [20, 50, 100, 150]
            batch_size = [1, 8, 16, 32]
            hu = [data_shape[-1], int(data_shape[-1] / 2.0), 20, 50, 100, 200, 300]
            param_grid = dict(
                epochs = epochs,
                batch_size = batch_size,
                hu = hu
            )

            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=folds, verbose=1)
            grid_result = grid.fit(view, target_labels)

            # summarize results
            print("View: %s" % view_names[view_idx])
            print("View: %s" % view_names[view_idx], file=output_file)
            print("Best: %f %s using %s" % (grid_result.best_score_, scoring, grid_result.best_params_))
            print("Best: %f %s using %s" % (grid_result.best_score_, scoring, grid_result.best_params_), file=output_file)
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
                print("%f (%f) with: %r" % (mean, stdev, param), file=output_file)
