from keras import backend as K

import keras.models
import numpy as np
import os
import sklearn.metrics as sklearn_metrics
import time
import tools.machine_learning as ml


def cross_val_score(model_generator, X, Y, batch_size=None, epochs=1, verbose=1, cv=10, scoring="accuracy", plot="", **kwargs):

    Y_o = Y[0]
    for y in Y:
        if not (y == Y_o).all():
            raise RuntimeError
    folds = ml.check_cv(cv, Y_o)

    results = []
    model_data = model_generator(**kwargs)
    if type(model_data) is not tuple:
        print("Model generator doesn't return a name. Returning model generator function as name.")
        model = model_data
        name = str(model_generator).split(" ")[1]
    else:
        model, name = model_data
    if plot != "":
        from keras.utils import plot_model
        plot_model(model, to_file=os.path.join(plot, '%s.png' % name),
                   show_shapes=True, show_layer_names=False)
    initial_weights = model.get_weights()
    if K.backend() == 'tensorflow':
        K.clear_session()
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
        results.append(result)
    print("Done  %s out of  %s | elapsed: %smin finished" % (
        len(folds), len(folds), sec_2_minutes(time.time() - beginning)
    ))
    if K.backend() == 'tensorflow':
        K.clear_session()
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
