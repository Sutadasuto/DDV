def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import multiprocessing
import numpy as np
import os
import math
import types

from copy import deepcopy
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import sklearn.model_selection as model
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score

import tools.arff_and_matrices as am


def check_cv(cv, labels, seed=10):
    num_instances = len(labels)
    if type(cv) is int:
        if cv < 2:
            print("Data must be splitted into 2 folds at least")
            raise ValueError
        elif cv > num_instances:
            print("Data can be splitted into more folds than instances.")
            raise ValueError
        else:
            indices = [i for i in range(len(labels))]
            np.random.RandomState(seed).shuffle(indices)
            step = int(math.floor(len(indices) / cv))
            packs = []

            for fold in range(cv):
                pack = []
                for position in range(step):
                    index = fold * step + position
                    try:
                        pack.append(indices[index])
                    except:
                        break
                packs.append(pack)
            if (index + 1) < len(indices):
                for position in range(index + 1, len(indices)):
                    packs[-1].append(indices[position])
            while len(packs[-1]) > int(math.ceil(len(indices) / cv)):
                for i in range(len(packs[-1]) - int(math.ceil(len(indices) / cv))):
                    if i < (cv - 1):
                        packs[i].append(packs[-1].pop())

            training_sets = []
            test_sets = []
            for fold in range(cv):
                training_instances = []

                for index in range(cv):
                    if index == fold:
                        test_instances = packs[index]
                    else:
                        training_instances += packs[index]
                training_sets.append(sorted(training_instances))
                test_sets.append(sorted(test_instances))

        folds = [() for i in range(cv)]
        for i in range(cv):
            folds[i] = (np.array(training_sets[i]), np.array(test_sets[i]))
        return folds

    elif type(cv) is list:
        for pair in cv:
            for element in pair[0]:
                if element in pair[1]:
                    print("Test instances cannot contain instances from training.")
                    raise ValueError
            for element in pair[1]:
                if element in pair[0]:
                    print("Training instances cannot contain instances from test.")
                    raise ValueError
            indices = np.concatenate((pair[0], pair[1]))
            indices.sort()
            if len(indices) == len(labels):
                for i in range(len(labels)):
                    if not i in indices:
                        print("An instance's index was not found in a fold")
                        raise ValueError
            else:
                print("The number of indices in the fold did not match the number of instances")
                raise ValueError
        return cv


def chi2_evaluation(arffInput):

    samples, classLabels, relation, attributeNames = am.arff_to_nparray(arffInput)
    resultMatrix = np.array(["Attribute"] + attributeNames).transpose()
    scores = np.concatenate((np.array(["Mutual Information"]), chi2(samples, classLabels))).transpose()
    resultMatrix = np.column_stack((resultMatrix, scores))
    return  resultMatrix


def coincident_failure_diversity(matrix):

    array = np.array(matrix)
    data = array[2:-2, 1:-1].astype(float)

    bestExpectedResult = []
    N = len(data[0,:])
    M = len(data)
    m = [0] * (N+1)
    for row in data:
        wrong_views = N - np.sum(row)
        m[int(wrong_views)]+=1
        bestExpectedResult.append(np.max(row))
    maximum_possible_accuracy = round(np.sum(bestExpectedResult) / len(bestExpectedResult), 3) * 100
    p = [0] * (N+1)
    for i in range(len(m)):
        p[i] = float(m[i])/M

    if p[0] == 1:
        cfd = 0
    elif p[0] < 1:
        sum = 0
        for n in range(1,N):
            sum += ((float(N-n))/(float(N-1))) * p[n]
        cfd = sum/(1-p[0])

    return maximum_possible_accuracy, cfd


def complementarity_analysis(classifier, databasesFolder=None, modalityFiles=None, exceptions=[], folds=None, showProba=None):

    if databasesFolder == None:
        databasesFolder = "datasets"
    if folds == None:
        folds = 10
    if showProba == None:
        showProba = False
    if len(exceptions) == 0:
        exceptions = ["early_fusion.arff", "syntax_informed.arff"]
    if modalityFiles == None:
        modalityFiles = sorted([os.path.join(databasesFolder, f) for f in os.listdir(databasesFolder)
                        if os.path.isfile(os.path.join(databasesFolder, f))
                        and not f.startswith('.') and f[-5:].lower() == ".arff" and not f in exceptions],
                       key=lambda f: f.lower())
    else:
        modalityFiles = [os.path.join(databasesFolder, f) for f in modalityFiles]
    try:
        with open (os.path.join(databasesFolder, "list_of_instances.csv")) as listOfInstances:
            instanceNames = listOfInstances.readlines()
    except:
        print("There was an error reading the list of evaluated instances.")
        raise
    instanceNames = [name.strip() for name in instanceNames]
    if folds == len(instanceNames):
        from sklearn.model_selection import LeaveOneOut
        indices = [i for i in range(len(instanceNames))]
        folds = LeaveOneOut().split(indices)
    instanceNames += ["Accuracy", "AUC"]

    resultMatrix = np.array([[str(classifier).split("(")[0], ""] + instanceNames]).transpose()
    for arffFile in modalityFiles:
        matrix, labels, relation, attributes = am.arff_to_nparray(arffFile)
        classes = list(set(labels))
        classes.sort()
        print("\n" + str(classifier).split("(")[0])
        print("Relation: " + relation)
        predictedLabels = model.cross_val_predict(classifier, matrix, labels, cv=folds, n_jobs=multiprocessing.cpu_count())
        if isinstance(folds, types.GeneratorType):
            folds = LeaveOneOut().split(indices)
        if showProba:
            try:
                probabilities = model.cross_val_predict(classifier, matrix, labels, method='predict_proba', cv=folds, n_jobs=multiprocessing.cpu_count())
                if isinstance(folds, types.GeneratorType):
                    folds = LeaveOneOut().split(indices)
            except:
                probabilities = np.array([[0, 0] for i in range(len(labels))])
                binary = preprocessing.label_binarize(predictedLabels, classes=list(reversed(classes)))
                for i in range(len(binary)):
                    if binary[i, 0] == 0:
                        tuple = [0, 1]
                    else:
                        tuple = [1, 0]
                    probabilities[i] = np.array(tuple)

        print(confusion_matrix(labels, predictedLabels))
        # accuracy = round(metrics.accuracy_score(labels, predictedLabels)*100, 1)
        accuracy = cross_val_score(classifier, matrix, labels, cv=folds, scoring="accuracy", n_jobs=multiprocessing.cpu_count())
        accuracy = round(accuracy.mean() * 100, 1)
        if isinstance(folds, types.GeneratorType):
            folds = LeaveOneOut().split(indices)
        #print metrics.accuracy_score(labels, model.cross_val_predict(classifier, matrix, labels, cv=10))
        #auc = metrics.roc_auc_score(preprocessing.label_binarize(labels, classes=list(reversed(classes))),
        #                            preprocessing.label_binarize(predictedLabels, classes=list(reversed(classes))))
        #auc = round(auc, 3)
        try:
            auc = cross_val_score(classifier, matrix, preprocessing.label_binarize(labels, classes), cv=folds, scoring="roc_auc", n_jobs=multiprocessing.cpu_count())
            auc = round(auc.mean(), 3)
        except:
            print("AUC cannot be calculated")
            auc = 0
        print("Accuracy: %s\nAUC: %s" % (accuracy, auc))
        if isinstance(folds, types.GeneratorType):
            folds = LeaveOneOut().split(indices)
        #print metrics.roc_auc_score(preprocessing.label_binarize(labels, classes=list(reversed(classes))),
        #                            preprocessing.label_binarize(model.cross_val_predict(classifier, matrix, labels, cv=10),
        #                                                         classes=list(reversed(classes))))
        if showProba:
            newColumn = np.concatenate((np.array([["", ""], [classes[0] + " probability", classes[1] + " probability"]]),
                                        probabilities, np.array([["", ""], ["", ""]])))
            resultMatrix = np.column_stack((resultMatrix, newColumn))
        newColumn = np.array([np.concatenate((np.array([relation, "Guess"]), predictedLabels == labels,
                                              np.array([accuracy]), np.array([auc])))])
        resultMatrix = np.column_stack((resultMatrix, newColumn.transpose()))
    newColumn = np.array([np.concatenate((np.array(["", "Real Label"]), labels,
                                          np.array(["", ""])))])
    resultMatrix = np.column_stack((resultMatrix, newColumn.transpose()))
    resultMatrix[resultMatrix == "True"] = "1"
    resultMatrix[resultMatrix == "False"] = "0"
    return resultMatrix


def concatenate_result_matrices(matrices):

    tuple = (matrices[0][:, :-1],)
    for matrix in matrices[1:]:
        tuple += (matrix[:,1:-1],)
    tuple += (matrices[-1][:,-1],)
    matrix = np.column_stack(tuple)
    return matrix


def evaluate_single_features(classifier, arffInput, folds=None):

    if folds == None:
        folds = 10

    matrix, Y, relation, attributes = am.arff_to_nparray(arffInput)
    if folds == len(Y):
        from sklearn.model_selection import LeaveOneOut
        folds = LeaveOneOut().split(matrix)
    classes = list(set(Y))
    resultMatrix = np.array([["", "Attributes"] + attributes]).transpose()
    scores = np.array([[str(classifier).split("(")[0], ""],["Accuracy", "AUC"]] + [["0","0"] for i in range(len(attributes))])
    for i in range(len(attributes)):
        X = matrix[:,i].reshape(-1,1)
        predictedLabels = model.cross_val_predict(classifier, X, Y, cv=folds, n_jobs=multiprocessing.cpu_count())
        # accuracy = round(metrics.accuracy_score(Y, predictedLabels), 3) * 100
        accuracy = cross_val_score(classifier, X, Y, cv=folds, scoring="accuracy", n_jobs=multiprocessing.cpu_count())
        if isinstance(folds, types.GeneratorType):
            folds = LeaveOneOut().split(matrix)
        # auc = metrics.roc_auc_score(preprocessing.label_binarize(Y, classes),
        #                             preprocessing.label_binarize(predictedLabels, classes))
        # auc = round(auc, 3)
        try:
            auc = cross_val_score(classifier, X, preprocessing.label_binarize(Y, classes), cv=folds, scoring="roc_auc", n_jobs=multiprocessing.cpu_count())
            auc = round(auc.mean(), 3)
        except:
            print("AUC cannot be calculated")
            auc = 0
        if isinstance(folds, types.GeneratorType):
            folds = LeaveOneOut().split(matrix)
        scores[i+2] = np.array([str(accuracy), str(auc)])
    resultMatrix = np.column_stack((resultMatrix, scores))
    return  resultMatrix


def f_evaluation(arffInput):

    samples, classLabels, relation, attributeNames = am.arff_to_nparray(arffInput)
    resultMatrix = np.array(["Attribute"] + attributeNames).transpose()
    scores = np.concatenate((np.array(["ANOVA F-value"]), f_classif(samples, classLabels)[0])).transpose()
    resultMatrix = np.column_stack((resultMatrix, scores))
    return resultMatrix


def get_complementarity_summary(matrices):

    summaryMatrix = matrices[0][2:-2,0]
    labels = matrices[0][2:-2,-1]
    newColumns = []
    header = ["Instance"]
    classifiers = []

    for matrix in matrices:
        classifiers.append(matrix[0,0])
        dataColumn = []
        data = matrix[2:-2,1:-1].astype(float)
        for row in range(len(data)):
            dataColumn.append(round(np.sum(data[row])/len(data[row]), 2))
        newColumns.append(np.vstack(np.array(dataColumn)))
    newColumns = np.column_stack(tuple(newColumns))
    averageColumn = []
    for row in range(len(newColumns)):
        averageColumn.append(round(np.sum(newColumns[row])/len(newColumns[row]), 2))
    newColumns = np.column_stack((newColumns, np.vstack(np.array(averageColumn))))
    summaryMatrix = np.column_stack((summaryMatrix, newColumns, labels))

    header += classifiers
    header += ["Average", "Label"]
    summaryMatrix = np.concatenate((np.array([header]), summaryMatrix))
    return summaryMatrix


def hard_majority_vote_evaluation(classifier, databasesFolder=None, modalityFiles=None, folds=None, relationName=None):

    if databasesFolder == None:
        databasesFolder = "datasets"
    if folds == None:
        folds = 10
    if relationName == None:
        relationName = "majority_vote"
    if modalityFiles == None:
        modalityFiles = sorted([os.path.join(databasesFolder, f) for f in os.listdir(databasesFolder)
                        if os.path.isfile(os.path.join(databasesFolder, f))
                        and not f.startswith('.') and f[-5:].lower() == ".arff"],
                       key=lambda f: f.lower())
    else:
        modalityFiles = [os.path.join(databasesFolder, f) for f in modalityFiles]
    try:
        with open (os.path.join(databasesFolder, "list_of_instances.csv")) as listOfInstances:
            instanceNames = listOfInstances.readlines()
    except:
        print("There was an error reading the list of evaluated instances.")
        raise
    print("\nMethod: " + relationName)
    instanceNames = [name.strip() for name in instanceNames]
    instanceNames += ["Accuracy", "AUC"]

    resultMatrix = np.array([[str(classifier).split("(")[0], ""] + instanceNames]).transpose()
    matrix, labels, relation, attributes = am.arff_to_nparray(modalityFiles[0])
    folds = check_cv(folds, labels)
    final_labels = ["None" for i in range(len(labels))]
    accuracy = []
    auc = []
    for pair in folds:
        predictionLists = []
        for arffFile in modalityFiles:
            matrix, labels, relation, attributes = am.arff_to_nparray(arffFile)
            classes = list(set(labels))
            classes.sort()
            classifier.fit(matrix[pair[0]], labels[pair[0]])
            predictionLists.append(classifier.predict(matrix[pair[1]]))
            # predictionLists.append(model.cross_val_predict(classifier, matrix, labels, cv=folds))
        predictedLabels = []
        for instance in range(len(predictionLists[0])):
            votes = [modality[instance] for modality in predictionLists]
            maxVoted = 0
            for classLabel in classes:
                classVotes = votes.count(classLabel)
                if classVotes > maxVoted:
                    maxVoted = classVotes
                    winner = classLabel
            predictedLabels.append(winner)
        for idx, value in enumerate(predictedLabels):
            position = pair[1][idx]
            final_labels[position] = value
        predictedLabels = np.array(predictedLabels)
        accuracy.append(metrics.accuracy_score(labels[pair[1]], predictedLabels))
        try:
            auc.append(metrics.roc_auc_score(preprocessing.label_binarize(labels[pair[1]], classes=list(reversed(classes))),
                                    preprocessing.label_binarize(predictedLabels, classes=list(reversed(classes)))))
        except:
            print("AUC cannot be calculated")
            auc.append(0)

    # accuracy = round(metrics.accuracy_score(labels, predictedLabels)*100, 1)
    # print metrics.accuracy_score(labels, model.cross_val_predict(classifier, matrix, labels, cv=10))
    # auc = metrics.roc_auc_score(preprocessing.label_binarize(labels, classes=list(reversed(classes))),
    #                             preprocessing.label_binarize(predictedLabels, classes=list(reversed(classes))))
    # auc = round(auc, 3)
    accuracy = np.array(accuracy)
    auc= np.array(auc)
    accuracy = round(accuracy.mean() * 100, 1)
    auc = round(auc.mean(), 3)
    print("Accuracy: %s\nAUC: %s" % (accuracy, auc))
    #print metrics.roc_auc_score(preprocessing.label_binarize(labels, classes=list(reversed(classes))),
    #                            preprocessing.label_binarize(model.cross_val_predict(classifier, matrix, labels, cv=10),
    #                                                         classes=list(reversed(classes))))
    final_labels = np.array(final_labels)
    newColumn = np.array([np.concatenate((np.array([relationName, "Guess"]), final_labels == labels,
                                          np.array([accuracy]), np.array([auc])))])
    resultMatrix = np.column_stack((resultMatrix, newColumn.transpose()))
    newColumn = np.array([np.concatenate((np.array(["","Real Label"]), labels,
                                          np.array(["", ""])))])
    resultMatrix = np.column_stack((resultMatrix, newColumn.transpose()))
    resultMatrix[resultMatrix == "True"] = "1"
    resultMatrix[resultMatrix == "False"] = "0"
    return resultMatrix


def mutual_information_evaluation(arffInput):

    samples, classLabels, relation, attributeNames = am.arff_to_nparray(arffInput)
    resultMatrix = np.array(["Attribute"] + attributeNames).transpose()
    scores = np.concatenate((np.array(["Mutual Information"]), mutual_info_classif(samples, classLabels))).transpose()
    resultMatrix = np.column_stack((resultMatrix, scores))
    return resultMatrix


def my_method(object, databases_folder=None, folds=None, relation_name=None, plots=False):

    object.set_plots_path(os.path.join(object.get_plots_path(), str(object.booster).split("(")[0]))
    if databases_folder == None:
        databases_folder = "datasets"
    if folds == None:
        folds = 10
    if relation_name == None:
        relation_name = "our_method"

    try:
        with open (os.path.join(databases_folder, "list_of_instances.csv")) as listOfInstances:
            instanceNames = listOfInstances.readlines()
    except:
        print("There was an error reading the list of evaluated instances.")
        raise

    print("\nMethod: " + relation_name)
    instanceNames = [name.strip() for name in instanceNames]
    instanceNames += ["Accuracy", "AUC"]

    result_matrix = np.array([[str(object.booster).split("(")[0], ""] + instanceNames]).transpose()
    # predicted_labels = object.cross_val_predict(folds, plots, plots, plots)
    accuracy, auc, predicted_labels = object.cross_val_score(folds, plots, plots, plots)

    # accuracy = round(metrics.accuracy_score(object.labels, predicted_labels)*100, 1)
    # auc = metrics.roc_auc_score(preprocessing.label_binarize(object.labels, classes=list(reversed(object.classes))),
    #                             preprocessing.label_binarize(predicted_labels, classes=list(reversed(object.classes))))
    # auc = round(auc, 3)
    accuracy = round(accuracy.mean()*100, 1)
    auc = round(auc.mean(), 3)
    print(confusion_matrix(object.labels, predicted_labels))
    print("Accuracy: %s\nAUC: %s" % (accuracy, auc))
    new_column = np.array([np.concatenate((np.array([relation_name, "Guess"]), predicted_labels == object.labels,
                                          np.array([accuracy]), np.array([auc])))])
    result_matrix = np.column_stack((result_matrix, new_column.transpose()))
    new_column = np.array([np.concatenate((np.array(["", "Real Label"]), object.labels,
                                          np.array(["", ""])))])
    result_matrix = np.column_stack((result_matrix, new_column.transpose()))
    result_matrix[result_matrix == "True"] = "1"
    result_matrix[result_matrix == "False"] = "0"
    return result_matrix


def proba_majority_vote_evaluation(classifier, databasesFolder=None, modalityFiles=None, folds=None, relationName=None):

    if databasesFolder == None:
        databasesFolder = "datasets"
    if folds == None:
        folds = 10
    if relationName == None:
        relationName = "proba_majority_vote"
    if modalityFiles == None:
        modalityFiles = sorted([os.path.join(databasesFolder, f) for f in os.listdir(databasesFolder)
                        if os.path.isfile(os.path.join(databasesFolder, f))
                        and not f.startswith('.') and f[-5:].lower() == ".arff"],
                       key=lambda f: f.lower())
    else:
        modalityFiles = [os.path.join(databasesFolder, f) for f in modalityFiles]
    try:
        with open (os.path.join(databasesFolder, "list_of_instances.csv")) as listOfInstances:
            instanceNames = listOfInstances.readlines()
    except:
        print("There was an error reading the list of evaluated instances.")
        raise
    print("\nMethod: " + relationName)
    instanceNames = [name.strip() for name in instanceNames]
    instanceNames += ["Accuracy", "AUC"]

    resultMatrix = np.array([[str(classifier).split("(")[0], ""] + instanceNames]).transpose()
    matrix, labels, relation, attributes = am.arff_to_nparray(modalityFiles[0])
    folds = check_cv(folds, labels)
    final_labels = ["None" for i in range(len(labels))]
    accuracy = []
    auc = []
    for pair in folds:
        predictionLists = []
        for arffFile in modalityFiles:
            matrix, labels, relation, attributes = am.arff_to_nparray(arffFile)
            classes = list(set(labels))
            classes.sort()
            classifier.fit(matrix[pair[0]], labels[pair[0]])
            try:
                predictionLists.append(classifier.predict_proba(matrix[pair[1]]))
            except:
                probabilities = np.array([[0, 0] for i in range(len(labels[pair[1]]))])
                binary = preprocessing.label_binarize(classifier.predict(matrix[pair[1]]),
                                                      classes=list(reversed(classes)))
                for i in range(len(binary)):
                    if binary[i, 0] == 0:
                        couple = [0, 1]
                    else:
                        couple = [1, 0]
                    probabilities[i] = np.array(couple)
                predictionLists.append(probabilities)

        predictedLabels = []
        for instance in range(len(predictionLists[0])):
            votes = np.array([modality[instance] for modality in predictionLists])
            total_votes = np.sum(votes, axis=0)
            winner = classes[np.argmax(total_votes)]
            predictedLabels.append(winner)
        for idx, value in enumerate(predictedLabels):
            position = pair[1][idx]
            final_labels[position] = value
        predictedLabels = np.array(predictedLabels)
        accuracy.append(metrics.accuracy_score(labels[pair[1]], predictedLabels))
        try:
            auc.append(metrics.roc_auc_score(preprocessing.label_binarize(labels[pair[1]], classes=list(reversed(classes))),
                                    preprocessing.label_binarize(predictedLabels, classes=list(reversed(classes)))))
        except:
            print("AUC cannot be calculated")
            auc.append(0)

    # accuracy = round(metrics.accuracy_score(labels, predictedLabels)*100, 1)
    # print metrics.accuracy_score(labels, model.cross_val_predict(classifier, matrix, labels, cv=10))
    # auc = metrics.roc_auc_score(preprocessing.label_binarize(labels, classes=list(reversed(classes))),
    #                             preprocessing.label_binarize(predictedLabels, classes=list(reversed(classes))))
    # auc = round(auc, 3)
    accuracy = np.array(accuracy)
    auc= np.array(auc)
    accuracy = round(accuracy.mean() * 100, 1)
    auc = round(auc.mean(), 3)
    print("Accuracy: %s\nAUC: %s" % (accuracy, auc))
    #print metrics.roc_auc_score(preprocessing.label_binarize(labels, classes=list(reversed(classes))),
    #                            preprocessing.label_binarize(model.cross_val_predict(classifier, matrix, labels, cv=10),
    #                                                         classes=list(reversed(classes))))
    final_labels = np.array(final_labels)
    newColumn = np.array([np.concatenate((np.array([relationName, "Guess"]), final_labels == labels,
                                          np.array([accuracy]), np.array([auc])))])
    resultMatrix = np.column_stack((resultMatrix, newColumn.transpose()))
    newColumn = np.array([np.concatenate((np.array(["","Real Label"]), labels,
                                          np.array(["", ""])))])
    resultMatrix = np.column_stack((resultMatrix, newColumn.transpose()))
    resultMatrix[resultMatrix == "True"] = "1"
    resultMatrix[resultMatrix == "False"] = "0"
    return resultMatrix


def stacking_evaluation(classifier, databasesFolder=None, modalityFiles=None, folds=None, relationName=None):

    if databasesFolder == None:
        databasesFolder = "datasets"
    if folds == None:
        folds = 10
    if relationName == None:
        relationName = "stacking"
    if modalityFiles == None:
        modalityFiles = sorted([os.path.join(databasesFolder, f) for f in os.listdir(databasesFolder)
                        if os.path.isfile(os.path.join(databasesFolder, f))
                        and not f.startswith('.') and f[-5:].lower() == ".arff"],
                       key=lambda f: f.lower())
    else:
        modalityFiles = [os.path.join(databasesFolder, f) for f in modalityFiles]
    try:
        with open (os.path.join(databasesFolder, "list_of_instances.csv")) as listOfInstances:
            instanceNames = listOfInstances.readlines()
    except:
        print("There was an error reading the list of evaluated instances.")
        raise
    print("\nMethod: " + relationName)
    instanceNames = [name.strip() for name in instanceNames]
    instanceNames += ["Accuracy", "AUC"]

    resultMatrix = np.array([[str(classifier).split("(")[0], ""] + instanceNames]).transpose()
    matrix, labels, relation, attributes = am.arff_to_nparray(modalityFiles[0])
    folds = check_cv(folds, labels)
    stacker = deepcopy(classifier)
    final_labels = ["None" for i in range(len(labels))]
    accuracy = []
    auc = []
    for pair in folds:
        predictionLists = []
        test_predictions = []
        views = []
        for arffFile in modalityFiles:
            matrix, labels, relation, attributes = am.arff_to_nparray(arffFile)
            views.append(relation)
            classes = list(set(labels))
            classes.sort()
            classifier.fit(matrix[pair[0]], labels[pair[0]])
            prediction = classifier.predict(matrix[pair[0]])
            # prediction = model.cross_val_predict(classifier, matrix[pair[0]], labels[pair[0]], cv=10)
            prediction = prediction.reshape(-1,1)
            test_prediction = classifier.predict(matrix[pair[1]])
            test_prediction = test_prediction.reshape(-1,1)
            predictionLists.append(preprocessing.label_binarize(prediction, neg_label=-1, classes=list(reversed(classes))))
            test_predictions.append(preprocessing.label_binarize(test_prediction, neg_label=-1, classes=list(reversed(classes))))
        newMatrix = np.column_stack(tuple(predictionLists))
        new_test_matrix = np.column_stack(tuple(test_predictions))
        stacker.fit(newMatrix, labels[pair[0]])
        predictedLabels = stacker.predict(new_test_matrix)
        # predictedLabels = model.cross_val_predict(classifier, newMatrix, labels, cv=folds)
        for idx, value in enumerate(predictedLabels):
            position = pair[1][idx]
            final_labels[position] = value
        accuracy.append(metrics.accuracy_score(labels[pair[1]], predictedLabels))
        try:
            auc.append(metrics.roc_auc_score(preprocessing.label_binarize(labels[pair[1]], classes=list(reversed(classes))),
                                             preprocessing.label_binarize(predictedLabels,
                                                                          classes=list(reversed(classes)))))
        except:
            print("AUC cannot be calculated")
            auc.append(0)
    # accuracy = round(metrics.accuracy_score(labels, predictedLabels)*100, 1)
    #print metrics.accuracy_score(labels, model.cross_val_predict(classifier, matrix, labels, cv=10))
    # auc = metrics.roc_auc_score(preprocessing.label_binarize(labels, classes=list(reversed(classes))),
    #                             preprocessing.label_binarize(predictedLabels, classes=list(reversed(classes))))
    # auc = round(auc, 3)
    accuracy = np.array(accuracy)
    auc = np.array(auc)
    accuracy = round(accuracy.mean() * 100, 1)
    auc = round(auc.mean(), 3)
    print("Accuracy: %s\nAUC: %s" % (accuracy, auc))
    #print metrics.roc_auc_score(preprocessing.label_binarize(labels, classes=list(reversed(classes))),
    #                            preprocessing.label_binarize(model.cross_val_predict(classifier, matrix, labels, cv=10),
    #                                                         classes=list(reversed(classes))))
    final_labels = np.array(final_labels)
    newColumn = np.array([np.concatenate((np.array([relationName, "Guess"]), final_labels == labels,
                                          np.array([accuracy]), np.array([auc])))])
    resultMatrix = np.column_stack((resultMatrix, newColumn.transpose()))
    newColumn = np.array([np.concatenate((np.array(["","Real Label"]), labels,
                                          np.array(["", ""])))])
    resultMatrix = np.column_stack((resultMatrix, newColumn.transpose()))
    resultMatrix[resultMatrix == "True"] = "1"
    resultMatrix[resultMatrix == "False"] = "0"
    return resultMatrix


def stacking_proba_evaluation(classifier, databasesFolder=None, modalityFiles=None, folds=None, relationName=None):

    if databasesFolder == None:
        databasesFolder = "datasets"
    if folds == None:
        folds = 10
    if relationName == None:
        relationName = "stacking_proba"
    if modalityFiles == None:
        modalityFiles = sorted([os.path.join(databasesFolder, f) for f in os.listdir(databasesFolder)
                        if os.path.isfile(os.path.join(databasesFolder, f))
                        and not f.startswith('.') and f[-5:].lower() == ".arff"],
                       key=lambda f: f.lower())
    else:
        modalityFiles = [os.path.join(databasesFolder, f) for f in modalityFiles]
    try:
        with open (os.path.join(databasesFolder, "list_of_instances.csv")) as listOfInstances:
            instanceNames = listOfInstances.readlines()
    except:
        print("There was an error reading the list of evaluated instances.")
        raise
    print("\nMethod: " + relationName)
    instanceNames = [name.strip() for name in instanceNames]
    instanceNames += ["Accuracy", "AUC"]

    resultMatrix = np.array([[str(classifier).split("(")[0], ""] + instanceNames]).transpose()
    matrix, labels, relation, attributes = am.arff_to_nparray(modalityFiles[0])
    folds = check_cv(folds, labels)
    stacker = deepcopy(classifier)
    final_labels = ["None" for i in range(len(labels))]
    accuracy = []
    auc = []
    for pair in folds:
        predictionLists = []
        test_predictions = []
        views = []
        for arffFile in modalityFiles:
            matrix, labels, relation, attributes = am.arff_to_nparray(arffFile)
            views.append(relation)
            classes = list(set(labels))
            classes.sort()
            prediction = []
            test_prediction = []
            classifier.fit(matrix[pair[0]], labels[pair[0]])
            try:
                probabilities = classifier.predict_proba(matrix[pair[0]])
                # probabilities = model.cross_val_predict(classifier, matrix, labels, method='predict_proba', cv=folds)
                test_probability = classifier.predict_proba(matrix[pair[1]])
            except:
                probabilities = np.array([[0, 0] for i in range(len(labels[pair[0]]))])
                binary = preprocessing.label_binarize(classifier.predict(matrix[pair[0]]),
                                                      classes=list(reversed(classes)))
                for i in range(len(binary)):
                    if binary[i, 0] == 0:
                        couple = [0, 1]
                    else:
                        couple = [1, 0]
                    probabilities[i] = np.array(couple)

                test_probability = np.array([[0, 0] for i in range(len(labels[pair[1]]))])
                binary = preprocessing.label_binarize(classifier.predict(matrix[pair[1]]),
                                                      classes=list(reversed(classes)))
                for i in range(len(binary)):
                    if binary[i, 0] == 0:
                        couple = [0, 1]
                    else:
                        couple = [1, 0]
                    test_probability[i] = np.array(couple)

            for couple in probabilities:
                labelIndex = couple.tolist().index(max(couple))
                if labelIndex == 0:
                    prediction.append(float(couple[labelIndex]))
                elif labelIndex == 1:
                    prediction.append(float(-couple[labelIndex]))

            for couple in test_probability:
                labelIndex = couple.tolist().index(max(couple))
                if labelIndex == 0:
                    test_prediction.append(float(couple[labelIndex]))
                elif labelIndex == 1:
                    test_prediction.append(float(-couple[labelIndex]))

            prediction = np.array(prediction).reshape(-1,1)
            predictionLists.append(prediction)
            test_prediction = np.array(test_prediction).reshape(-1,1)
            test_predictions.append(test_prediction)
        newMatrix = np.column_stack(tuple(predictionLists))
        new_test_matrix = np.column_stack(tuple(test_predictions))
        stacker.fit(newMatrix, labels[pair[0]])
        predictedLabels = stacker.predict(new_test_matrix)
        for idx, value in enumerate(predictedLabels):
            position = pair[1][idx]
            final_labels[position] = value
        accuracy.append(metrics.accuracy_score(labels[pair[1]], predictedLabels))
        try:
            auc.append(metrics.roc_auc_score(preprocessing.label_binarize(labels[pair[1]], classes=list(reversed(classes))),
                                             preprocessing.label_binarize(predictedLabels,
                                                                          classes=list(reversed(classes)))))
        except:
            print("AUC cannot be calculated")
            auc.append(0)
    # predictedLabels = model.cross_val_predict(classifier, newMatrix, labels, cv=folds)
    # accuracy = round(metrics.accuracy_score(labels, predictedLabels)*100, 1)
    #print metrics.accuracy_score(labels, model.cross_val_predict(classifier, matrix, labels, cv=10))
    # auc = metrics.roc_auc_score(preprocessing.label_binarize(labels, classes=list(reversed(classes))),
    #                             preprocessing.label_binarize(predictedLabels, classes=list(reversed(classes))))
    # auc = round(auc, 3)
    accuracy = np.array(accuracy)
    auc = np.array(auc)
    accuracy = round(accuracy.mean() * 100, 1)
    auc = round(auc.mean(), 3)
    print("Accuracy: %s\nAUC: %s" % (accuracy, auc))
    #print metrics.roc_auc_score(preprocessing.label_binarize(labels, classes=list(reversed(classes))),
    #                            preprocessing.label_binarize(model.cross_val_predict(classifier, matrix, labels, cv=10),
    #                                                         classes=list(reversed(classes))))
    final_labels = np.array(final_labels)
    newColumn = np.array([np.concatenate((np.array([relationName, "Guess"]), final_labels == labels,
                                          np.array([accuracy]), np.array([auc])))])
    resultMatrix = np.column_stack((resultMatrix, newColumn.transpose()))
    newColumn = np.array([np.concatenate((np.array(["","Real Label"]), labels,
                                          np.array(["", ""])))])
    resultMatrix = np.column_stack((resultMatrix, newColumn.transpose()))
    resultMatrix[resultMatrix == "True"] = "1"
    resultMatrix[resultMatrix == "False"] = "0"
    return resultMatrix