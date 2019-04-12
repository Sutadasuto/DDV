from keras import backend as K

import keras.models
import numpy as np
import sklearn.metrics as metrics
import time
import tools.machine_learning as ml


def cross_val_score(model, X, Y, cv=10, scoring="accuracy", **kwargs):

    Y_o = Y[0]
    for y in Y:
        if not (y == Y_o).all():
            raise RuntimeError
    folds = ml.check_cv(cv, Y_o)

    results = []
    beginning = time.time()
    for fold_num, fold in enumerate(folds, 1):
        now = time.time()
        fold_model = keras.models.clone_model(model)
        fold_model.set_weights(model.get_weights())
        fold_model.compile(loss=model.loss, optimizer=model.optimizer, metrics=model.metrics)
        X_train = []
        for x in X:
            X_train.append(x[fold[0]])
        Y_train = Y_o[fold[0]]
        X_test = []
        for x in X:
            X_test.append(x[fold[1]])
        Y_test = Y_o[fold[1]]

        fold_model.fit(X_train, Y_train, **kwargs)
        print("Done  %s out of  %s | elapsed: %smin finished" % (fold_num, len(folds), sec_2_minutes(time.time() - now)))
        now = time.time()
        preds = fold_model.predict(X_test)
        Y_pred = np.where(preds > 0.5, 1, 0)
        print("%s/%s instances predicted in %s sec." % (len(Y_pred), len(Y_pred), round(time.time()-now, 2)))
        if scoring == "accuracy":
            result = metrics.accuracy_score(Y_test, Y_pred)
        elif scoring == "roc_auc":
            result = metrics.roc_auc_score(Y_test, Y_pred)
        results.append(result)
        if K.backend() == 'tensorflow':
            K.clear_session()
    print("Done  %s out of  %s | elapsed: %smin finished" % (
        len(folds), len(folds), sec_2_minutes(time.time() - beginning)
    ))
    return np.array(results)


def sec_2_minutes(seconds):
    min = int(seconds/60)
    sec = seconds % 60
    minutes = str(min + round(sec/60,1))
    return minutes


def write_result(results, label, output_file):
    output = "%s: %.2f%% (%.2f%%)\n" % (label, results.mean() * 100, results.std() * 100)
    print(output)
    output_file.write(output)
