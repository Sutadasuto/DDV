from keras import backend as K

import keras.models
import numpy as np
import os
import sklearn.metrics as sklearn_metrics
import time
import tools.machine_learning as ml


def cross_val_score(model_generator, X, Y, feat_standardization=False, cv=10,
                    scoring="accuracy", plot="", batch_size=None, epochs=1, verbose=1, **kwargs):

    if type(Y) is list:
        Y_o = Y[0]
        for y in Y:
            if not (y == Y_o).all():
                raise RuntimeError
    else:
        Y_o = Y
    folds = ml.check_cv(cv, Y_o)

    results = []
    model_data = model_generator(**kwargs)
    if type(model_data) is not tuple:
        print("Model generator doesn't return a name. Returning model generator function as name.")
        model = model_data
        name = str(model_generator).split(" ")[1]
    else:
        model, name = model_data
    print("Validating %s" % (name))
    if plot != "":
        from keras.utils import plot_model
        plot_model(model, to_file=os.path.join(plot, '%s.png' % name),
                   show_shapes=True, show_layer_names=False)
    initial_weights = model.get_weights()
    if K.backend() == 'tensorflow':
        K.clear_session()
        del model
    beginning = time.time()
    for fold_num, fold in enumerate(folds, 1):
        now = time.time()

        model_data = model_generator(**kwargs)
        if type(model_data) is not tuple:
            model = model_data
        else:
            model = model_data[0]
        model.set_weights(initial_weights)
        X_train = []
        for x in X:
            X_train.append(x[fold[0]])
        Y_train = Y_o[fold[0]]
        X_test = []
        for x in X:
            X_test.append(x[fold[1]])
        Y_test = Y_o[fold[1]]

        if feat_standardization:
            standarizer = Standarizer()
            standarizer.fit(X_train)
            X_train = standarizer.transform(X_train)
            X_test = standarizer.transform(X_test)

        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)
        print("Done  %s out of  %s | elapsed: %smin finished" % (fold_num, len(folds), sec_2_minutes(time.time() - now)))
        now = time.time()
        preds = model.predict(X_test)
        Y_pred = np.where(preds > 0.5, 1, 0)
        print("%s/%s instances predicted in %s sec." % (len(Y_pred), len(Y_pred), round(time.time()-now, 2)))
        if scoring == "accuracy":
            result = sklearn_metrics.accuracy_score(Y_test, Y_pred)
        elif scoring == "roc_auc":
            result = sklearn_metrics.roc_auc_score(Y_test, Y_pred)
        print("%s: %s" % (scoring, round(result, 4)))
        results.append(result)
    print("Done  %s out of  %s | elapsed: %smin finished" % (
        len(folds), len(folds), sec_2_minutes(time.time() - beginning)
    ))
    if K.backend() == 'tensorflow':
        K.clear_session()
        del model
    return np.array(results), name


def sec_2_minutes(seconds):
    min = int(seconds/60)
    sec = seconds % 60
    minutes = str(min + round(sec/60,1))
    return minutes


def write_result(results, label, output_file):
    output = "%s: %.2f%% (%.2f%%)\n" % (label, results.mean() * 100, results.std() * 100)
    print(output)
    output_file.write(output)


class Standarizer():

    def __init__(self):
        self.means = []
        self.stds = []

    def fit(self, data):

        for view in data:
            n_features = view.shape[-1]
            view_means = []
            view_stds = []

            for feature in range(n_features):
                data_series = np.reshape(view[..., feature], (-1,))
                view_means.append(np.mean(data_series))
                view_stds.append(np.std(data_series))
            self.means.append(view_means)
            self.stds.append(view_stds)

    def inverse_transform(self, data):

        n_views = len(data)
        if n_views != len(self.means):
            raise ValueError

        for idx_view, view in enumerate(data):
            n_features = view.shape[-1]
            if n_features != len(self.means[idx_view]):
                raise ValueError

            for idx_feature in range(n_features):
                mean = self.means[idx_view][idx_feature]
                std = self.stds[idx_view][idx_feature]
                if std != 0:
                    view[..., idx_feature] *= std
                    view[..., idx_feature] += mean
        return data

    def transform(self, data):

        n_views = len(data)
        if n_views != len(self.means):
            raise ValueError

        for idx_view, view in enumerate(data):
            n_features = view.shape[-1]
            if n_features != len(self.means[idx_view]):
                raise ValueError

            for idx_feature in range(n_features):
                mean = self.means[idx_view][idx_feature]
                std = self.stds[idx_view][idx_feature]
                if std != 0:
                    view[..., idx_feature] -= mean
                    view[..., idx_feature] /= std
        return data