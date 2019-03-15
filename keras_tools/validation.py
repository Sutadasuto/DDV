from sklearn.model_selection import StratifiedKFold

import copy
import keras.models
import numpy as np
import sklearn.metrics as metrics
import tools.machine_learning as ml


def cross_val_score(model, X, Y, cv=10, scoring="accuracy", **kwargs):

    Y_o = Y[0]
    for y in Y:
        if not (y == Y_o).all():
            raise RuntimeError
    folds = ml.check_cv(cv, Y_o)

    results = []
    for fold in folds:
        # fold_model = copy.deepcopy(model)
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
        preds = fold_model.predict(X_test)
        Y_pred = np.where(preds > 0.5, 1, 0)
        if scoring == "accuracy":
            result = metrics.accuracy_score(Y_test, Y_pred)
        elif scoring == "roc_auc":
            result = metrics.roc_auc_score(Y_test, Y_pred)
        results.append(result)

    return np.array(results)
