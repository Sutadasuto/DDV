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
import keras
import keras_tools.validation as metrics
import keras_tools.sequences as sequences
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
    model.add(Dense(output, kernel_initializer="normal", activation="sigmoid"))
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
    model.add(Dense(output, kernel_initializer="normal", activation="sigmoid"))
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
    model.add(Dense(output, kernel_initializer="normal", activation="sigmoid"))
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
    model.add(Dense(output, kernel_initializer="normal", activation="sigmoid"))
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
    dense = Dense(output, kernel_initializer="normal", activation="sigmoid")(lstm)
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
    dense_2 = Dense(output, kernel_initializer="normal", activation="sigmoid")(dense_1)
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
    dense = Dense(output, kernel_initializer="normal", activation="sigmoid")(result)
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
    dense = Dense(output, kernel_initializer="normal", activation="sigmoid")(result)
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
    dense = Dense(output, kernel_initializer="normal", activation="sigmoid")(cat)
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
    dense_2 = Dense(output, kernel_initializer="normal", activation="sigmoid")(dense_1)
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
    dense = Dense(output, kernel_initializer="normal", activation="sigmoid")(cat)
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
    dense = Dense(output, kernel_initializer="normal", activation="sigmoid")(cat)
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
    dense = Dense(output, kernel_initializer="normal", activation="sigmoid")(video_representation)
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
    model.add(Dense(2, kernel_initializer="normal", activation="linear"))
    model.add(Dense(1, kernel_initializer="normal", activation="linear"))
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
                  epochs=100, batch_size=16, gpu=True):
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
        folds = [fold for fold in LeaveOneOut().split(a)]
    else:
        folds = cv

    with open(os.path.join(output_folder, "lstm_results_modalities_%s_%s.txt" % (seq_reduction, reduction)), "w+") as output_file:
        for stream_idx in range(len(inputs)):
            classifiers = []
            classifiers.append(KerasClassifier(build_fn=create_basic_lstm, hu=hu, timesteps=X[stream_idx].shape[1],
                                               data_dim=X[stream_idx].shape[2], output=1, dropout=dropout, gpu=gpu,
                                               epochs=epochs, batch_size=batch_size, verbose=2))
            classifiers.append(KerasClassifier(build_fn=create_basic_lstm, hu=hu, timesteps=X[stream_idx].shape[1],
                                               data_dim=X[stream_idx].shape[2], output=1, dropout=dropout, gpu=gpu,
                                               epochs=epochs, batch_size=batch_size, verbose=2))
            classifiers.append(KerasClassifier(build_fn=create_attention_lstm, hu=hu, timesteps=X[stream_idx].shape[1],
                                               data_dim=X[stream_idx].shape[2], output=1, dropout=dropout, gpu=gpu,
                                               epochs=epochs, batch_size=batch_size, verbose=2))
            classifiers.append(
                KerasClassifier(build_fn=create_attention_context_lstm, hu=hu, timesteps=X[stream_idx].shape[1],
                                data_dim=X[stream_idx].shape[2], output=1, dropout=dropout, gpu=gpu,
                                epochs=epochs, batch_size=batch_size, verbose=2))
            header = "Database: %s\nData: %s\nHidden units: %s, Epochs: %s, Batch Size: %s, Dropout: %s, Seq. reduction: %s, %s\n" % (
                os.path.split(inputs[stream_idx])[0], os.path.split(inputs[stream_idx])[1], hu, epochs, batch_size,
                dropout, seq_reduction, reduction)
            print(header.strip())
            output_file.write(header)
            labels = ["Basic", "Double dense", "Attention", "Attention with context"]
            for idx, classifier in enumerate(classifiers):
                results = cross_val_score(classifier, X[stream_idx], Y[stream_idx], scoring="roc_auc", cv=folds,
                                          verbose=1)
                metrics.write_result(results, labels[idx], output_file)
            output_file.write("\n")

        # Multidata
        classifiers = []
        classifiers.append(
            create_multidata_basic_lstm(hu=hu, input_shapes=input_shapes, output=1, dropout=dropout, gpu=gpu))
        classifiers.append(
            create_multidata_basic_lstm(hu=hu, input_shapes=input_shapes, output=1, dropout=dropout, gpu=gpu))
        classifiers.append(
            create_multidata_attention_lstm(hu=hu, input_shapes=input_shapes, output=1, dropout=dropout, gpu=gpu))
        classifiers.append(
            create_multidata_attention_context_lstm(hu=hu, input_shapes=input_shapes, output=1, dropout=dropout,
                                                    gpu=gpu))

        streams = [os.path.split(i)[1] for i in inputs]
        header = "Database: %s\nData: %s\nHidden units: %s, Epochs: %s, Batch Size: %s, Dropout: %s, Seq. reduction: %s, %s\n" % (
            os.path.split(inputs[0])[0], " + ".join(streams), hu, epochs, batch_size,
            dropout, seq_reduction, reduction)
        print(header.strip())
        output_file.write(header)
        labels = ["Early fusion Basic", "Early fusion Double dense", "Early fusion Attention",
                  "Early fusion Attention with context"]
        for idx, classifier in enumerate(classifiers):
            results = metrics.cross_val_score(classifier, X, Y, scoring="roc_auc", cv=folds, epochs=epochs,
                                              batch_size=batch_size, verbose=2)
            metrics.write_result(results, labels[idx], output_file)
        output_file.write("\n")

        # Multistream
        classifiers = []
        classifiers.append(
            create_multistream_basic_lstm(hu=hu, input_shapes=input_shapes, output=1, dropout=dropout, gpu=gpu))
        classifiers.append(
            create_multistream_basic_lstm(hu=hu, input_shapes=input_shapes, output=1, dropout=dropout, gpu=gpu))
        classifiers.append(
            create_multistream_attention_lstm(hu=hu, input_shapes=input_shapes, output=1, dropout=dropout, gpu=gpu))
        classifiers.append(
            create_multistream_attention_context_lstm(hu=hu, input_shapes=input_shapes, output=1, dropout=dropout,
                                                      gpu=gpu))

        header = "Database: %s\nData: %s\nHidden units: %s, Epochs: %s, Batch Size: %s, Dropout: %s, Seq. reduction: %s, %s\n" % (
            os.path.split(inputs[0])[0], " + ".join(streams), hu, epochs, batch_size,
            dropout, seq_reduction, reduction)
        print(header.strip())
        output_file.write(header)
        labels = ["Late fusion Basic", "Late fusion Double dense", "Late fusion Attention",
                  "Late fusion Attention with context"]
        for idx, classifier in enumerate(classifiers):
            results = metrics.cross_val_score(classifier, X, Y, scoring="roc_auc", cv=folds, epochs=epochs,
                                              batch_size=batch_size, verbose=2)
            metrics.write_result(results, labels[idx], output_file)
        output_file.write("\n")

def my_method(modalities, cv=10, seq_reduction="padding", reduction="avg", output_folder=None, hu=50, dropout=None,
              epochs=100, batch_size=16, gpu=True):
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
        folds = [fold for fold in LeaveOneOut().split(a)]
    else:
        folds = cv

    architectures = [architecture_1, architecture_2]
    models = []
    names = []
    streams = [", ".join([os.path.split(i)[1] for i in modality]) for modality in modalities]
    from keras.utils import plot_model
    for architecture in architectures:
        model, name = architecture(input_shapes, hu, 1, dropout, gpu)
        plot_model(model, to_file=os.path.join(output_folder, '%s_%s_%s.png' % (name, seq_reduction, reduction)),
                   show_shapes=True, show_layer_names=False)
        models.append(model)
        names.append(name)
    with open(os.path.join(output_folder, "%s_%s_%s.txt" % (name, seq_reduction, reduction)), "w+") as output_file:
        header = "Database: %s\nData: %s\nHidden units: %s, Epochs: %s, Batch Size: %s, Dropout: %s, Seq. reduction: %s, %s\n" % (
            os.path.split(os.path.split(modalities[0][0])[0])[0], " + ".join(streams), hu, epochs, batch_size,
            dropout, seq_reduction, reduction)
        print(header.strip())
        output_file.write(header)
        for model, name in zip(models, names):
            results = metrics.cross_val_score(model, X, Y, scoring="roc_auc", cv=folds, epochs=epochs,
                                              batch_size=batch_size, verbose=2)
            print(header.strip())
            metrics.write_result(results, name, output_file)