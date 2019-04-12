import keras
import keras.backend as K
from keras.models import Model
from keras.layers import LSTM, Input, Flatten, Dense

import numpy as np


def triplet_loss_function():

    a=0


def get_triplets(x, y):

    labels = list(set(y))

    instance_sets = []

    for label in labels:
        instance_sets.append(list(np.where(y==label)[0]))

    triplets = []
    for label in range(len(instance_sets)):
        same_class_pairs = []
        for instance_1 in range(len(instance_sets[label]) - 1):
            i = instance_sets[label][instance_1]
            for instance_2 in range(instance_1 + 1, len(instance_sets[label])):
                j = instance_sets[label][instance_2]
                same_class_pairs.append([i,j])
        for pair in same_class_pairs:
            for instance in instance_sets[label + (-1)**label]:
                triplets.append(pair + [instance])

    X = []
    for triplet in triplets:
        list_4d = []
        for instance in triplet:
            list_4d.append(x[instance, :, :])
        X.append(np.array(list_4d))

    return np.array(X)


def create_multistream_model(X, rnn_generator, **kwargs):

    # Inspired from: https://github.com/keras-team/keras/issues/10333
    # Another reference: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
    seq_shape = (X.shape[2], X.shape[3])
    n = X.shape[1]
    # Define a shared model
    input_seq = Input(seq_shape)
    layer_n = rnn_generator(**kwargs)(input_seq)
    # Create model
    shared_model = Model(input_seq, layer_n, name="shared_rnn")

    # Define n input sequences
    seq = []
    encoded = []
    for i in range(n):
        seq_i = Input(seq_shape)
        seq.append(seq_i)
        encoded.append(shared_model(seq_i))

    # Let's do classification for fun
    cat = keras.layers.concatenate(encoded, axis=-1)
    softmax = Dense(1, activation="softmax")(cat)

    model = Model(seq, softmax)
    Y = np.zeros((X.shape[0], model.layers[-1].output_shape))

    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)