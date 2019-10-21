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
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder

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
import sklearn.metrics as metrics
import tools.arff_and_matrices as am
import tools.machine_learning as ml


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


def create_encoding_lstm(hu, time_steps, data_dim, output, dropout=0, gpu=True):

    if K.backend() == 'tensorflow':
        K.clear_session()
    # create model
    model = Sequential()
    if gpu:
        model.add(CuDNNLSTM(hu, return_sequences=False, input_shape=(time_steps, data_dim), name="lstm"))
    else:
        model.add(LSTM(hu, return_sequences=False, input_shape=(time_steps, data_dim), name="lstm"))
    # model.add(Attention())
    model.add(Dropout(dropout, seed=0, name="dropout1"))
    hu2 = max(1, int(hu/2))
    # model.add(Dense(hu2, activation="sigmoid", name="fc1"))
    model.add(Dense(output, activation="sigmoid", name="fc2"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


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

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    X = [item for sublist in X for item in sublist]
    Y = [item for sublist in Y for item in sublist]
    if type(cv) is int:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10)
        folds = [fold for fold in folds.split(X[0], Y[0])]
    elif cv == "loo":
        folds = [fold for fold in LeaveOneOut().split(X[0])]
    else:
        folds = cv

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

            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=folds, verbose=100)
            grid_result = grid.fit(view, target_labels)
            if K.backend() == 'tensorflow':
                K.clear_session()
                del model

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


def views_train_features(hyperparameters_file, modalities, seq_reduction="padding", reduction="avg", output_folder=None,
                         verbose=2):

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
                X_idx, Y_idx, encoder = sequences.get_input_sequences(view, length, True)
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
        output_folder = os.path.join(os.getcwd(), "tuned_models")

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    X = [item for sublist in X for item in sublist]
    Y = [item for sublist in Y for item in sublist]

    with open(hyperparameters_file) as grid:

        import ast
        view_found = False
        lines = grid.readlines()
        parameters = dict()
        for line in lines:
            if line.startswith("View: "):
                view_name = line.strip().replace("View: ", "")
                parameters[view_name] = None
                view_found = True
            elif view_found == True:
                parameters_dict = line.split(" using ")[1]
                parameters[view_name] = ast.literal_eval(parameters_dict)
                view_found = False

    for view_idx, view in enumerate(X):
        target_labels = Y[view_idx]
        if len(target_labels.shape) == 1:
            output = 1
        else:
            output = target_labels.shape[-1]
        data_shape = view.shape
        data_dim = data_shape[-1]
        time_steps = data_shape[-2]

        hyperparameters = parameters[view_names[view_idx]]
        model_parameters = dict()
        training_parameters = dict()
        for parameter in hyperparameters.keys():
            if parameter != "batch_size" and parameter != "epochs" and parameter != "optimizer":
                model_parameters[parameter] = hyperparameters[parameter]
            else:
                training_parameters[parameter] = hyperparameters[parameter]

        model = create_encoding_lstm(time_steps=time_steps, data_dim=data_dim, output=output, **model_parameters)
        model.fit(view, target_labels, verbose=verbose, **training_parameters)
        # serialize model to JSON
        model_json = model.to_json()
        with open(os.path.join(output_folder, "%s.json" % view_names[view_idx]), "w+") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(os.path.join(output_folder, "%s.h5" % view_names[view_idx]))
        print("Saved %s model to disk" % view_names[view_idx])

        extractor = Model(model.input, model.get_layer("last_layer").output)
        extractor.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        features = [np.concatenate(
            (extractor.predict(np.expand_dims(instance, axis=0)).reshape(-1,), label.reshape(-1,)),
            axis=-1
        ).tolist()
                    for instance, label
                    in zip(view, encoder.inverse_transform(target_labels))]

        header = [["HU_%s" % num for num in range(1, len(features[0]))] + ["Class"]]
        matrix = header + features
        am.create_arff(matrix, encoder.classes_, output_folder, view_names[view_idx], view_names[view_idx] + "_lstm")


def views_train_features_cv(hyperparameters_file, modalities, cv=10, seq_reduction="padding", reduction="avg", output_folder=None,
                         verbose=2):

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
                X_idx, Y_idx, encoder = sequences.get_input_sequences(view, length, True)
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
        output_folder = os.path.join(os.getcwd(), "tuned_models")

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    X = [item for sublist in X for item in sublist]
    Y = [item for sublist in Y for item in sublist]

    if type(cv) is int:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10)
        folds = [fold for fold in folds.split(X[0], Y[0])]
    elif cv == "loo":
        folds = [fold for fold in LeaveOneOut().split(X[0])]
    else:
        folds = cv

    with open(hyperparameters_file) as grid:

        import ast
        view_found = False
        lines = grid.readlines()
        parameters = dict()
        for line in lines:
            if line.startswith("View: "):
                view_name = line.strip().replace("View: ", "")
                parameters[view_name] = None
                view_found = True
            elif view_found == True:
                parameters_dict = line.split(" using ")[1]
                parameters[view_name] = ast.literal_eval(parameters_dict)
                view_found = False

    for view_idx, view in enumerate(X):
        target_labels = Y[view_idx]
        if len(target_labels.shape) == 1:
            output = 1
        else:
            output = target_labels.shape[-1]
        data_shape = view.shape
        data_dim = data_shape[-1]
        time_steps = data_shape[-2]

        hyperparameters = parameters[view_names[view_idx]]
        model_parameters = dict()
        training_parameters = dict()
        for parameter in hyperparameters.keys():
            if parameter != "batch_size" and parameter != "epochs" and parameter != "optimizer":
                model_parameters[parameter] = hyperparameters[parameter]
            else:
                training_parameters[parameter] = hyperparameters[parameter]

        features = [np.array([]) for n in range(data_shape[0])]
        for pair in folds:
            x_train = view[pair[0]]
            x_test = view[pair[1]]
            y_train = target_labels[pair[0]]
            y_test = target_labels[pair[1]]

            model = create_encoding_lstm(time_steps=time_steps, data_dim=data_dim, output=output, **model_parameters)
            model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=verbose, **training_parameters)

            extractor = Model(model.input, model.get_layer("last_layer").output)
            extractor.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            test_features = [np.concatenate(
                (extractor.predict(np.expand_dims(instance, axis=0)).reshape(-1,), label.reshape(-1,)),
                axis=-1
            ).tolist()
                        for instance, label
                        in zip(x_test, encoder.inverse_transform(y_test))]

            for idx, instance in enumerate(test_features):
                features[pair[1][idx]] = instance

            del model, extractor

        header = [["HU_%s" % num for num in range(1, len(features[0]))] + ["Class"]]
        matrix = header + features
        am.create_arff(matrix, encoder.classes_, output_folder, view_names[view_idx], view_names[view_idx] + "_lstm")


def cross_val_score(hyperparameters_file, modalities, clf, cv=10, scoring="accuracy", seq_reduction="padding",
                    reduction="avg", verbose=2):

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
                X_idx, Y_idx, encoder = sequences.get_input_sequences(view, length, True)
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

    X = [item for sublist in X for item in sublist]
    Y = [item for sublist in Y for item in sublist]

    if type(cv) is int:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10)
        folds = [fold for fold in folds.split(X[0], Y[0])]
    elif cv == "loo":
        folds = [fold for fold in LeaveOneOut().split(X[0])]
    else:
        folds = cv

    with open(hyperparameters_file) as grid:

        import ast
        view_found = False
        lines = grid.readlines()
        parameters = dict()
        for line in lines:
            if line.startswith("View: "):
                view_name = line.strip().replace("View: ", "")
                parameters[view_name] = None
                view_found = True
            elif view_found == True:
                parameters_dict = line.split(" using ")[1]
                parameters[view_name] = ast.literal_eval(parameters_dict)
                view_found = False

    matrix = [["View", scoring]]
    for view_idx, view in enumerate(X):
        target_labels = Y[view_idx]
        if len(target_labels.shape) == 1:
            output = 1
        else:
            output = target_labels.shape[-1]
        data_shape = view.shape
        data_dim = data_shape[-1]
        time_steps = data_shape[-2]

        hyperparameters = parameters[view_names[view_idx]]
        model_parameters = dict()
        training_parameters = dict()
        for parameter in hyperparameters.keys():
            if parameter != "batch_size" and parameter != "epochs" and parameter != "optimizer":
                model_parameters[parameter] = hyperparameters[parameter]
            else:
                training_parameters[parameter] = hyperparameters[parameter]

        scores = []
        print("View: %s" % view_names[view_idx])
        for pair in folds:
            x_train = view[pair[0]]
            x_test = view[pair[1]]
            y_train = target_labels[pair[0]]
            y_test = target_labels[pair[1]]

            model = create_encoding_lstm(time_steps=time_steps, data_dim=data_dim, output=output, **model_parameters)
            model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=verbose, **training_parameters)

            extractor = Model(model.input, model.get_layer("last_layer").output)
            extractor.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            train_features = extractor.predict(x_train)
            test_features = extractor.predict(x_test)

            clf.fit(train_features, y_train)
            y_pred = clf.predict_proba(test_features)

            if scoring == "roc_auc":
                y_pred = y_pred[...,-1]
                scores.append(metrics.roc_auc_score(y_test, y_pred))
            if scoring == "accuracy":
                y_pred = np.argmax(y_pred, axis=-1)
                scores.append(metrics.accuracy_score(y_test, y_pred))
            del model, extractor
        print("%s: %s" % (scoring, np.array(scores).mean()))
        matrix.append([view_names[view_idx], np.array(scores).mean()])

    with open("file.txt", "w+") as o:
        writer = csv.writer(o)
        writer.writerows(matrix)


def boosting_cross_val_score(hyperparameters_file, modalities, booster, stacker, cv=10, scoring="accuracy", seq_reduction="padding",
                    reduction="avg", verbose=2):

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
                X_idx, Y_idx, encoder = sequences.get_input_sequences(view, length, True)
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

    X = [item for sublist in X for item in sublist]
    Y = [item for sublist in Y for item in sublist]

    if type(cv) is int:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10)
        folds = [fold for fold in folds.split(X[0], Y[0])]
    elif cv == "loo":
        folds = [fold for fold in LeaveOneOut().split(X[0])]
    else:
        folds = cv

    with open(hyperparameters_file) as grid:

        import ast
        view_found = False
        lines = grid.readlines()
        parameters = dict()
        for line in lines:
            if line.startswith("View: "):
                view_name = line.strip().replace("View: ", "")
                parameters[view_name] = None
                view_found = True
            elif view_found == True:
                parameters_dict = line.split(" using ")[1]
                parameters[view_name] = ast.literal_eval(parameters_dict)
                view_found = False

    from tools import multimodal_fusion as fusion
    scores = []
    matrix = [["Method", scoring]]
    for pair in folds:
        train_datasets = []
        test_datasets = []
        train_labels = []
        test_labels = []
        view_scores = []
        for view_idx, view in enumerate(X):
            target_labels = Y[view_idx]
            if len(target_labels.shape) == 1:
                output = 1
            else:
                output = target_labels.shape[-1]
            data_shape = view.shape
            data_dim = data_shape[-1]
            time_steps = data_shape[-2]

            hyperparameters = parameters[view_names[view_idx]]
            model_parameters = dict()
            training_parameters = dict()
            for parameter in hyperparameters.keys():
                if parameter != "batch_size" and parameter != "epochs" and parameter != "optimizer":
                    model_parameters[parameter] = hyperparameters[parameter]
                else:
                    training_parameters[parameter] = hyperparameters[parameter]

            x_train = view[pair[0]]
            x_test = view[pair[1]]
            y_train = target_labels[pair[0]]
            y_test = target_labels[pair[1]]

            model = create_encoding_lstm(time_steps=time_steps, data_dim=data_dim, output=output, **model_parameters)
            model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=verbose, **training_parameters)

            feature_layer = "lstm"
            extractor = Model(model.input, model.get_layer(feature_layer).output)
            extractor.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            train_features = extractor.predict(x_train)
            train_datasets.append(train_features)
            test_features = extractor.predict(x_test)
            test_datasets.append(test_features)

            booster.fit(train_features, y_train)
            y_pred = booster.predict_proba(test_features)
            if scoring == "roc_auc":
                y_real = np.array([[0 for i in range(len(set(y_test)))] for j in range(len(y_test))])
                for idx, instance in enumerate(y_test):
                    y_real[idx][instance] = 1
                roc_auc = metrics.roc_auc_score(y_real, y_pred, average=None)
                view_scores.append(roc_auc[0])
            if scoring == "accuracy":
                y_pred = np.argmax(y_pred, axis=-1)
                view_scores.append(metrics.accuracy_score(y_test, y_pred))

        t_methods = [ml.early_fusion_from_numpy, ml.hard_majority_vote_from_numpy, ml.proba_majority_vote_from_numpy,
                   ml.stacking_from_numpy, ml.stacking_proba_from_numpy]
        arguments = [
            {"classifier": booster, "train_datasets": train_datasets, "train_labels": y_train, "validation_datasets": test_datasets},
            {"classifier": booster, "train_datasets": train_datasets, "train_labels": y_train, "validation_datasets": test_datasets},
            {"classifier": booster, "train_datasets": train_datasets, "train_labels": y_train, "validation_datasets": test_datasets},
            {"classifier": booster, "stacker": stacker, "train_datasets": train_datasets, "train_labels": y_train, "validation_datasets": test_datasets},
            {"classifier": booster, "stacker": stacker, "train_datasets": train_datasets, "train_labels": y_train, "validation_datasets": test_datasets}
        ]
        fusion_scores = []
        for idx, method in enumerate(t_methods):
            y_pred = method(**arguments[idx])
            if scoring == "roc_auc":
                y_real = np.array([[0 for i in range(len(set(y_test)))] for j in range(len(y_test))])
                for instance_idx, instance in enumerate(y_test):
                    y_real[instance_idx][instance] = 1
                roc_auc = metrics.roc_auc_score(y_real, y_pred, average=None)
                fusion_scores.append(roc_auc[0])
            if scoring == "accuracy":
                y_pred = np.argmax(y_pred, axis=-1)
                fusion_scores.append(metrics.accuracy_score(y_test, y_pred))

        methods = [fusion.BSSD_From_Numpy, fusion.S4DB_From_Numpy]
        arguments = [
            {"booster": booster, "modality": "multimodal", "databases": train_datasets, "labels": y_train, "dataset_names": view_names},
            {"booster": booster, "stacker": stacker, "modality": "multimodal", "databases": train_datasets, "labels": y_train, "dataset_names": view_names}
        ]
        subscores = []
        for idx, method in enumerate(methods):
            o = method(**arguments[idx])
            o.fit()
            y_pred = o.predict_proba(test_datasets)
            if scoring == "roc_auc":
                y_real = np.array([[0 for i in range(len(set(y_test)))] for j in range(len(y_test))])
                for instance_idx, instance in enumerate(y_test):
                    y_real[instance_idx][instance] = 1
                roc_auc = metrics.roc_auc_score(y_real, y_pred, average=None)
                subscores.append(roc_auc[0])
            if scoring == "accuracy":
                y_pred = np.argmax(y_pred, axis=-1)
                subscores.append(metrics.accuracy_score(y_test, y_pred))

        scores.append(view_scores + fusion_scores + subscores)
        del model, extractor

    scores = np.array(scores)
    for idx, method in enumerate(view_names + t_methods + methods):
        name = str(method)
        score = scores[:, idx].mean()
        matrix.append([name, score])
    with open("file.txt", "w+") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)


def create_hierarchical_encoding_lstm(input_shapes, view_dicts, output, gpu=True):

    seq = []
    lstms = []
    for view_idx, input_shape in enumerate(input_shapes):
        model_dict = view_dicts[view_idx]
        # Create a LSTM for each view
        input_data = Input(input_shape)
        if gpu:
            lstm = CuDNNLSTM(model_dict["hu"], input_shape=input_shape, return_sequences=False, name="lstm_%s" % view_idx)(input_data)
        else:
            lstm = LSTM(model_dict["hu"], input_shape=input_shape, return_sequences=False, name="lstm_%s" % view_idx)(input_data)
        lstm = Dense(output, activation="sigmoid", name="fc_%s" % view_idx)(lstm)
        seq.append(input_data)
        lstms.append(lstm)
    # Concatenate independent streams
    dense = keras.layers.concatenate(lstms)
    # reduction = 4
    # num = int(K.int_shape(dense)[-1] / reduction)
    # num = max(1, num)
    # num_fc = 1
    # while num > 1:
    #     dense = Dense(num, activation="sigmoid", name="fc_%s" % num_fc)(dense)
    #     num_fc += 1
    #     num = int(K.int_shape(dense)[-1] / reduction)
    #     num = max(1, num)
    dense = Dense(output, activation="sigmoid", name="fc_last")(dense)
    model = Model(seq, dense)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def hierarchical_cross_val_score(hyperparameters_file, modalities, cv=10, scoring="accuracy", seq_reduction="padding",
                    reduction="avg", verbose=2):

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
                X_idx, Y_idx, encoder = sequences.get_input_sequences(view, length, True)
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

    X = [item for sublist in X for item in sublist]
    Y = [item for sublist in Y for item in sublist]

    if type(cv) is int:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=10)
        folds = [fold for fold in folds.split(X[0], Y[0])]
    elif cv == "loo":
        folds = [fold for fold in LeaveOneOut().split(X[0])]
    else:
        folds = cv

    with open(hyperparameters_file) as grid:

        import ast
        view_found = False
        lines = grid.readlines()
        parameters = dict()
        for line in lines:
            if line.startswith("View: "):
                view_name = line.strip().replace("View: ", "")
                parameters[view_name] = None
                view_found = True
            elif view_found == True:
                parameters_dict = line.split(" using ")[1]
                parameters[view_name] = ast.literal_eval(parameters_dict)
                view_found = False

    scores = []
    matrix = [["Method", scoring]]
    for pair in folds:
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        input_shapes = []
        view_dicts = []
        for view_idx, view in enumerate(X):
            target_labels = Y[view_idx]
            if len(target_labels.shape) == 1:
                output = 1
            else:
                output = target_labels.shape[-1]
            data_shape = view.shape
            data_dim = data_shape[-1]
            time_steps = data_shape[-2]
            input_shapes.append((time_steps, data_dim))

            hyperparameters = parameters[view_names[view_idx]]
            model_parameters = dict()
            for parameter in hyperparameters.keys():
                if parameter != "batch_size" and parameter != "epochs" and parameter != "optimizer":
                    model_parameters[parameter] = hyperparameters[parameter]
            view_dicts.append(model_parameters)

            x_train.append(view[pair[0]])
            x_test.append(view[pair[1]])
            y_train.append(target_labels[pair[0]])
            y_test.append(target_labels[pair[1]])

        if type(y_train) is list:
            y_o = y_train[0]
            for y in y_train:
                if not (y == y_o).all():
                    raise RuntimeError
            y_train = y_train[0]
        else:
            y_train = y_train
        if type(y_test) is list:
            y_o = y_test[0]
            for y in y_test:
                if not (y == y_o).all():
                    raise RuntimeError
            y_test = y_test[0]
        else:
            y_test= y_test

        model = create_hierarchical_encoding_lstm(input_shapes, view_dicts, output)
        model.fit(x_train, y_train, 8, 300, verbose=verbose, validation_data=(x_test, y_test))
        y_pred = model.predict(x_test)
        if scoring == "roc_auc":
            roc_auc = metrics.roc_auc_score(y_test, y_pred, average=None)
            scores.append(roc_auc)
        if scoring == "accuracy":
            y_pred = np.where(y_pred > 0.5, 1, 0)
            scores.append(metrics.accuracy_score(y_test, y_pred))

    scores = np.array(scores)
    score = scores.mean()
    matrix.append(["Hierarchical", score])
    with open("file_lstm.txt", "w+") as f:
        writer = csv.writer(f)
        writer.writerows(matrix)
