from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras.models import Sequential
from keras.layers import Dense, LSTM, CuDNNLSTM, Activation, Dropout, TimeDistributed
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from keras_tools.attention_layers import Attention, AttentionWithContext
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import copy
import numpy as np
import os
import pandas


# define baseline model
def baseline_model(timesteps, data_dim, output, dropout=None, return_sequences=False, gpu=True):
    # expected input data shape: (batch_size, timesteps, data_dim)
    # create model
    #def prepare_model(timesteps, data_dim, num_classes):
    model = Sequential()
    if gpu:
        model.add(CuDNNLSTM(20, return_sequences=return_sequences,
                       input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    else:
        model.add(LSTM(20, return_sequences=return_sequences,
                            input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    # model.add(CuDNNLSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    # model.add(CuDNNLSTM(32))  # return a single vector of dimension 32
    # model.add(Dense(1, activation='sigmoid'))
    # compile model
    if dropout is float:
        model.add(Dropout(dropout))
    model.add(Dense(output, kernel_initializer="normal", activation="sigmoid"))
    # model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
    #return prepare_model(timesteps, data_dim, num_classes)


# define baseline model
def baseline_model_wrapped(timesteps, data_dim, output, dropout=None, return_sequences=False, gpu=True):
    # expected input data shape: (batch_size, timesteps, data_dim)
    # create model
    def prepare_model(timesteps, data_dim, output, dropout, return_sequences, gpu):
        model = Sequential()
        if gpu:
            model.add(CuDNNLSTM(20, return_sequences=return_sequences,
                                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
        else:
            model.add(LSTM(20, return_sequences=return_sequences,
                           input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
        # model.add(CuDNNLSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
        # model.add(CuDNNLSTM(32))  # return a single vector of dimension 32
        # model.add(Dense(1, activation='sigmoid'))
        # compile model
        if dropout is float:
            model.add(Dropout(dropout))
        model.add(Dense(output, kernel_initializer="normal", activation="sigmoid"))
        # model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    return prepare_model(timesteps, data_dim, output, dropout, return_sequences, gpu)


def basic_binary_lstm(input):
    # Input can be either a folder containing csv files or a .npy file
    if input.endswith(".npy"):
        loaded_array = np.load(input)
        X = loaded_array[0]
        Y = loaded_array[1]
    else:
        classes = sorted([f for f in os.listdir(input)
                          if os.path.isdir(os.path.join(input, f)) and not f.startswith('.')],
                         key=lambda f: f.lower())
        X = []
        Y = []
        sequence_lengths = []
        for class_name in classes:
            files = sorted([f for f in os.listdir(os.path.join(input, class_name))
                            if os.path.isfile(os.path.join(input, class_name, f)) and not f.startswith('.')
                            and f.endswith(".csv")], key=lambda f: f.lower())
            for file in files:
                df = pandas.read_csv(os.path.join(input, class_name, file))
                values = df.values
                nan_inds = np.where(np.isnan(values))
                values[nan_inds] = 0
                inf_inds = np.where(np.isinf(values))
                values[inf_inds] = np.sign(values[inf_inds])
                X.append(values)
                sequence_lengths.append(len(df.values))
                Y.append(class_name)
        sequence_lengths = np.array(sequence_lengths)
        avg_length = int(sum(sequence_lengths) / len(sequence_lengths))
        X = np.array(X)
        X = sequence.pad_sequences(X, maxlen=avg_length, dtype="float64")
        Y = np.array(Y)
        encoder = LabelEncoder()
        encoder.fit(Y)
        Y = encoder.transform(Y)
        # Y = np_utils.to_categorical(Y)
        # Y = [copy.deepcopy(Y.reshape(-1,1)) for i in range(X.shape[1])]
        # Y = np.array(Y)

    model = baseline_model(X.shape[1], X.shape[2], 1, None, False,
                           False)  # 1 for one class only (binary classification)
    model.summary()
    print("Initial parameters:")
    weights1 = model.layers[0].get_weights()
    print(weights1)
    pred_labels1 = model.predict_classes(X)
    for i in range(len(Y)):
        print("Predicted: %s, Real: %s" % (pred_labels1[i], Y[i]))
    a = 0
    model.fit(X, Y, epochs=50, batch_size=8, verbose=1, validation_split=0.1)
    print("Trained parameters:")
    weights2 = model.layers[0].get_weights()
    print(weights2)
    pred_labels2 = model.predict_classes(X)
    for i in range(len(Y)):
        print("Predicted: %s, Real: %s" % (pred_labels2[i], Y[i]))
    a = 0


def basic_binary_lstm_cv(input):
    # Input can be either a folder containing csv files or a .npy file
    if input.endswith(".npy"):
        loaded_array = np.load(input)
        X = loaded_array[0]
        Y = loaded_array[1]
    else:
        classes = sorted([f for f in os.listdir(input)
                          if os.path.isdir(os.path.join(input, f)) and not f.startswith('.')],
                         key=lambda f: f.lower())
        X = []
        Y = []
        sequence_lengths = []
        for class_name in classes:
            files = sorted([f for f in os.listdir(os.path.join(input, class_name))
                            if os.path.isfile(os.path.join(input, class_name, f)) and not f.startswith('.')
                            and f.endswith(".csv")], key=lambda f: f.lower())
            for file in files:
                df = pandas.read_csv(os.path.join(input, class_name, file))
                values = df.values
                nan_inds = np.where(np.isnan(values))
                values[nan_inds] = 0
                inf_inds = np.where(np.isinf(values))
                values[inf_inds] = np.sign(values[inf_inds])
                X.append(values)
                sequence_lengths.append(len(df.values))
                Y.append(class_name)
        sequence_lengths = np.array(sequence_lengths)
        avg_length = int(sum(sequence_lengths) / len(sequence_lengths))
        X = np.array(X)
        X = sequence.pad_sequences(X, maxlen=avg_length, dtype="float64")
        Y = np.array(Y)
        encoder = LabelEncoder()
        encoder.fit(Y)
        Y = encoder.transform(Y)
        # Y = np_utils.to_categorical(Y)
        # Y = [copy.deepcopy(Y.reshape(-1,1)) for i in range(X.shape[1])]
        # Y = np.array(Y)

    seed = 8

    classifier = KerasClassifier(build_fn=create_basic_lstm, timesteps=X.shape[1], data_dim=X.shape[2], output=1,
                                 dropout=None, gpu=False, epochs=50, batch_size=8, verbose=1)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(classifier, X, Y, cv=kfold, verbose=1)
    print("Result: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    # model.summary()
    # print("Initial parameters:")
    # weights1 = model.layers[0].get_weights()
    # print(weights1)
    # pred_labels1 = model.predict_classes(X)
    # for i in range(len(Y)):
    #     print("Predicted: %s, Real: %s" % (pred_labels1[i], Y[i]))
    # a = 0
    # model.fit(X, Y, epochs=50, batch_size=8, verbose=1, validation_split=0.1)
    # print("Trained parameters:")
    # weights2 = model.layers[0].get_weights()
    # print(weights2)
    # pred_labels2 = model.predict_classes(X)
    # for i in range(len(Y)):
    #     print("Predicted: %s, Real: %s" % (pred_labels2[i], Y[i]))
    # a = 0


def create_basic_lstm(timesteps=1, data_dim=1, output=1, dropout=None, gpu=True):
    # expected input data shape: (batch_size, timesteps, data_dim)
    # create model
    model = Sequential()
    if gpu:
        model.add(CuDNNLSTM(20, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    else:
        model.add(LSTM(20, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    if dropout is float:
        model.add(Dropout(dropout))
    model.add(Dense(output, kernel_initializer="normal", activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_model():
    model = Sequential()
    model.add(LSTM(20, return_sequences=False,
                        input_shape=(2799, 75)))  # returns a sequence of vectors of dimension 32
    model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_stacking_lstm2(timesteps=1, data_dim=1, output=1, dropout=None, gpu=True):
    # expected input data shape: (batch_size, timesteps, data_dim)
    # create model
    model = Sequential()
    if gpu:
        model.add(CuDNNLSTM(20, input_shape=(timesteps, data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 32
    else:
        model.add(LSTM(20, input_shape=(timesteps, data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 32
    if dropout is float:
        model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(output)))
    model.add(Dense(output, kernel_initializer="normal", activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def create_stacking_lstm(timesteps=1, data_dim=1, output=1, dropout=None, gpu=True):
    # expected input data shape: (batch_size, timesteps, data_dim)
    # create model
    model = Sequential()
    if gpu:
        model.add(CuDNNLSTM(20, input_shape=(timesteps, data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 32
    else:
        model.add(LSTM(20, input_shape=(timesteps, data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 32
    if dropout is float:
        model.add(Dropout(dropout))
    model.add(Attention())
    model.add(Dense(output, kernel_initializer="normal", activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def dense_binary_lstm(input):
    # Input can be either a folder containing csv files or a .npy file
    if input.endswith(".npy"):
        loaded_array = np.load(input)
        X = loaded_array[0]
        Y = loaded_array[1]
    else:
        classes = sorted([f for f in os.listdir(input)
                          if os.path.isdir(os.path.join(input, f)) and not f.startswith('.')],
                         key=lambda f: f.lower())
        X = []
        Y = []
        sequence_lengths = []
        for class_name in classes:
            files = sorted([f for f in os.listdir(os.path.join(input, class_name))
                            if os.path.isfile(os.path.join(input, class_name, f)) and not f.startswith('.')
                            and f.endswith(".csv")], key=lambda f: f.lower())
            for file in files:
                df = pandas.read_csv(os.path.join(input, class_name, file))
                X.append(df.values[:,1:])
                sequence_lengths.append(len(df.values))
                Y.append(class_name)
        sequence_lengths = np.array(sequence_lengths)
        avg_length = int(sum(sequence_lengths)/len(sequence_lengths))
        X = np.array(X)
        X = sequence.pad_sequences(X, maxlen=avg_length)
        Y = np.array(Y)
        encoder = LabelEncoder()
        encoder.fit(Y)
        Y = encoder.transform(Y)
        # Y = np_utils.to_categorical(Y)
        # Y = [copy.deepcopy(Y.reshape(-1,1)) for i in range(X.shape[1])]
        # Y = np.array(Y)
    
    model = baseline_model(X.shape[1], X.shape[2], 1, 0.5, False) # 1 for one class only (binary classification)
    model.summary()
    pred_labels = model.predict(X)
    for i in range(len(Y)):
        print("Predicted: %s, Real: %s" % (pred_labels[i], Y[i]))
    a=0
    model.fit(X, Y, epochs=5, batch_size=32, verbose=1, validation_split=0.0)
    pred_labels = model.predict(X)
    for i in range(len(Y)):
        print("Predicted: %s, Real: %s" % (pred_labels[i], Y[i]))
    a=0


def get_data(input):
    classes = sorted([f for f in os.listdir(input)
                      if os.path.isdir(os.path.join(input, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    X = []
    Y = []
    sequence_lengths = []
    for class_name in classes:
        files = sorted([f for f in os.listdir(os.path.join(input, class_name))
                        if os.path.isfile(os.path.join(input, class_name, f)) and not f.startswith('.')
                        and f.endswith(".csv")], key=lambda f: f.lower())
        for file in files:
            df = pandas.read_csv(os.path.join(input, class_name, file))
            values = df.values
            nan_inds = np.where(np.isnan(values))
            values[nan_inds] = 0
            inf_inds = np.where(np.isinf(values))
            values[inf_inds] = np.sign(values[inf_inds])
            X.append(values)
            sequence_lengths.append(len(df.values))
            Y.append(class_name)
    sequence_lengths = np.array(sequence_lengths)
    avg_length = int(sum(sequence_lengths) / len(sequence_lengths))
    X = np.array(X)
    X = sequence.pad_sequences(X, maxlen=avg_length, dtype="float64")
    Y = np.array(Y)
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    return X, Y


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
    
    
def stacked_h_binary_lstm(input):
    # Input can be either a folder containing csv files or a .npy file
    if input.endswith(".npy"):
        loaded_array = np.load(input)
        X = loaded_array[0]
        Y = loaded_array[1]
    else:
        X, Y = get_data(input)

    seed = 8

    classifier = KerasClassifier(build_fn=create_basic_lstm, timesteps=X.shape[1], data_dim=X.shape[2], output=1,
                                 dropout=None, gpu=False, epochs=50, batch_size=8, verbose=1)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(classifier, X, Y, cv=kfold, verbose=1)
    print("Result: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def standard_vs_binary(input):
    # Input can be either a folder containing csv files or a .npy file
    if input.endswith(".npy"):
        loaded_array = np.load(input)
        X = loaded_array[0]
        Y = loaded_array[1]
    else:
        X, Y = get_data(input)

    seed = 8

    classifier_basic = KerasClassifier(build_fn=create_basic_lstm, timesteps=X.shape[1], data_dim=X.shape[2], output=1,
                                 dropout=None, gpu=True, epochs=50, batch_size=32, verbose=1)
    classifier_stacking = KerasClassifier(build_fn=create_stacking_lstm, timesteps=X.shape[1], data_dim=X.shape[2], output=1,
                                       dropout=None, gpu=True, epochs=50, batch_size=32, verbose=1)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results_stacking = cross_val_score(classifier_stacking, X, Y, cv=kfold, verbose=1)#, n_jobs=-1)
    results_basic = cross_val_score(classifier_basic, X, Y, cv=kfold, verbose=1, n_jobs=-1)
    print("Result basic: %.2f%% (%.2f%%)" % (results_basic.mean() * 100, results_basic.std() * 100))
    print("Result stacking: %.2f%% (%.2f%%)" % (results_stacking.mean() * 100, results_stacking.std() * 100))