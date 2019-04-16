import keras
import keras.backend as K
from keras.models import Model
from keras.layers import LSTM, Input, Flatten, Dense

import numpy as np


def create_multistream_model(single_model_generator, classification_model=True, **kwargs):
    single_video_model = single_model_generator(**kwargs)
    if type(single_video_model) is tuple:
        name = "LMNN %s" % single_video_model[1]
        single_video_model = single_video_model[0]
    else:
        name = "LMNN %s" % str(single_model_generator).split(" ")[1]
    if classification_model:
        single_video_model = Model(single_video_model.input, single_video_model.layers[-2].output)

    n_samples = 3
    video_input_shapes = [K.int_shape(input_shape) for input_shape in single_video_model.input]
    model_input_shapes = []
    for input_shape in video_input_shapes:
        model_input_shape = []
        for dimension in input_shape:
            if dimension is not None:
                model_input_shape.append(dimension)
        model_input_shape[-1] *= n_samples
        model_input_shapes.append(model_input_shape)
        
    seq = []
    separated_video_features = []
    for shape in model_input_shapes:
        n_features = int(shape[-1] / n_samples)
        
        in_seq = Input(shape)
        seq.append(in_seq)

        vs = [keras.layers.Lambda(lambda x: x[..., n*n_features:(n+1)*n_features])(in_seq) for n in range(n_samples)]
        separated_video_features.append(vs)
    separated_video_features = np.array(separated_video_features)

    single_video_features = []
    for video in range(len(separated_video_features[0])):
        single_video_features.append(separated_video_features[:, video].tolist())

    fs = [single_video_model(features) for features in single_video_features]
    f = keras.layers.concatenate(fs)

    model = LMNN_Model(inputs=seq, outputs=f)
    model.compile(optimizer="adam", loss=model.triplet_loss_function, metrics=[model.triplet_loss_function])

    return model, name


class LMNN_Model(Model):

    def __init__(self, n_neighbors=5, beta=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.fs = None
        from sklearn.neighbors import KNeighborsClassifier
        self.knc = KNeighborsClassifier(n_neighbors=n_neighbors)
        a=0

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):

        triplets_indices = self.get_triplets_indices(y)
        train_data_generator = self.triplets_generator(x, y, triplets_indices, batch_size)
        H = super().fit_generator(train_data_generator, steps_per_epoch=len(triplets_indices) // batch_size,
                                  epochs=epochs, verbose=verbose,
                                  callbacks=callbacks, validation_data=validation_data,
                                  validation_steps=validation_steps,
                                  class_weight=class_weight, shuffle=shuffle, initial_epoch=initial_epoch)
        self.fs = self.layers[-2].predict(x)
        self.knc.fit(self.fs, y)

        return H

    def get_triplets_indices(self, y):

        if type(y) is list:
            y_o = y[0]
            for y in y:
                if not (y == y_o).all():
                    raise RuntimeError
        else:
            y_o = y
        labels = list(set(y_o))
        num_labels = len(labels)

        instance_sets = []

        for label in labels:
            instance_sets.append(list(np.where(y == label)[0]))

        triplets = []
        for label in range(num_labels):
            same_class_pairs = []
            for instance_1 in range(len(instance_sets[label]) - 1):
                i = instance_sets[label][instance_1]
                for instance_2 in range(instance_1 + 1, len(instance_sets[label])):
                    j = instance_sets[label][instance_2]
                    same_class_pairs.append([i, j])
            for pair in same_class_pairs:
                for instance in instance_sets[label + (-1) ** label]:
                    triplets.append([pair[0], pair[1]] + [instance])
                    triplets.append([pair[1], pair[0]] + [instance])

        return triplets

    def predict(self, x):

        fs = self.layers[-2].predict(x)
        return self.knc.predict(fs)

    def triplets_generator(self, x, y, triplets_indices, batch_size):

        idx = 0
        n_views = len(x)
        n_instances = len(y)
        while True:
            X = []
            labels = []

            while len(labels) < batch_size:
                if idx >= n_instances:
                    idx = 0
                new_instance = []
                triplet = triplets_indices[idx]
                for view in x:
                    feature_list = tuple([view[instance, :, :] for instance in triplet])
                    new_instance.append(np.concatenate(feature_list, axis=-1))
                X.append(new_instance)
                labels.append(y[triplet[0]])
                idx += 1

            batch_data = []
            for view in range(n_views):
                view_data = []
                for instance in range(batch_size):
                    view_data.append(X[instance][view])
                batch_data.append(np.array(view_data))

            yield (batch_data, np.array(labels))

    def triplet_loss_function(self, y_true, y_pred):
        n_samples = 3
        tensor_shape = K.int_shape(y_pred)
        n_features = int(tensor_shape[-1] / n_samples)
        fs = [y_pred[..., n * n_features:(n + 1) * n_features] for n in range(n_samples)]
        return self.triplet_loss(fs[0], fs[1], fs[2])

    def triplet_loss(self, f_is, f_js, f_ks):
        """
        Args:
          f_is: the embeddings for the X_i videos.
          f_js: the embeddings for the X_j videos.
          f_ks: the embeddings for the X_k videos.

        Returns:
          the triplet loss  as a float tensor.
        """

        pos_dist = K.sum(K.square(f_is - f_js), axis=-1)  # Euclidean distance to the square
        neg_dist = K.sum(K.square(f_is - f_ks), axis=-1)
        margin = K.constant(1.0)
        losses = pos_dist + self.beta * K.maximum(K.constant(0.0), margin + pos_dist - neg_dist)

        return K.mean(losses)
