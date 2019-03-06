import os
import csv
import numpy as np
import sys
import text.text_analysis as t
import tools.arff_and_matrices as am
import tools.subject_analysis as sa
import text.lib.dictionary as dictLib
import sklearn.base as base
import math
import time
import datetime
import sklearn.model_selection as model
import sklearn.metrics as metrics
import multiprocessing
from joblib import Parallel, delayed
from sklearn import preprocessing
from copy import deepcopy
import random
from tools import plots
from tools.machine_learning import check_cv as verify_folds


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def early_fusion(datasetsFolder, fileNames=None, exceptions=[], targetFileFolder=None, outputFileName=None,
                 relation=None):
    if outputFileName == None:
        outputFileName = "early_fusion"
    if targetFileFolder == None:
        targetFileFolder == datasetsFolder
    [mMatrix, classes, relationName] = am.arffs_to_matrix(datasetsFolder, fileNames, exceptions)
    if relation == None:
        relation = relationName
    am.create_arff(mMatrix, classes, targetFileFolder, outputFileName, relation)


def syntax_informed(textsFolder, accousticFolder, processedDataFolder=None, outputFileName=None, relationName=None,
                    sample_rate=None):
    if sample_rate == None:
        sample_rate = 0.01
    if processedDataFolder == None:
        processedDataFolder = "datasets"
    if outputFileName == None:
        outputFileName = "syntax_informed"
    if relationName == None:
        relationName = "syntax_informed_accoustic"

    tags = t.load_tags()
    classes = sorted([f for f in os.listdir(textsFolder)
                      if os.path.isdir(os.path.join(textsFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    headerFlag = True
    header = []
    matrix = []
    for className in classes:
        fileNames = sorted([f for f in os.listdir(os.path.join(textsFolder, className, "pos_timestamps"))
                            if os.path.isfile(os.path.join(textsFolder, className, "pos_timestamps", f))
                            and not f.startswith('.') and f[-4:].lower() == ".csv"],
                           key=lambda f: f.lower())
        for name in fileNames:
            resultingVector = []
            with open(os.path.join(textsFolder, className, "pos_timestamps", name)) as syntaxFile:
                line = syntaxFile.readline()
            if len(line) > 0:
                posStamps = line.split(",")
            else:
                posStamps = []
            with open(os.path.join(accousticFolder, className, name)) as accousticFile:
                reader = csv.reader(accousticFile)
                accousticMatrix = []
                initialTime = -sample_rate
                endTime = 0
                for row in reader:
                    accousticMatrix.append([initialTime, endTime] + row)
                    initialTime += sample_rate
                    endTime += sample_rate
            for tag in tags:
                if headerFlag:
                    for feature in accousticMatrix[0][2:-1]:
                        header.append("%s_%s" % (tag, feature))
                listOfTimes = []
                for pos in posStamps:
                    elements = pos.split(";")
                    if elements[0].lower() == tag.lower():
                        listOfTimes.append([float(time) for time in elements[1:]])
                index = 0
                rows = []
                for window in listOfTimes:
                    for i in range(index + 1, len(accousticMatrix)):
                        if accousticMatrix[i][0] >= window[0]:
                            j = i
                            while window[1] > accousticMatrix[j][1]:
                                rows.append(accousticMatrix[j])
                                j += 1
                                if j >= len(accousticMatrix):
                                    break
                            index = j - 1
                            break
                if len(rows) > 0:
                    values = [[float(number) for number in row[2:]] for row in rows]
                else:
                    values = [[0.0] * len(accousticMatrix[0][2:-1])]
                array = np.array(values)
                for i in range(len(accousticMatrix[0][2:-1])):
                    column = array[:, i]
                    average = np.sum(column) / len(column)
                    if np.isinf(average):
                        resultingVector.append(np.sign(average))
                    elif np.isnan(average):
                        resultingVector.append(0)
                    else:
                        resultingVector.append(average)
            matrix.append(resultingVector + [className])
            headerFlag = False
    header = header + ["Class"]
    matrix = [header] + matrix
    am.create_arff(matrix, classes, processedDataFolder, outputFileName, relationName)
    print("Syntax informed representation acquired.")


def au_informed(ofFeaturesFolder, accousticFolder, processedDataFolder=None, outputFileName=None, relationName=None,
                sample_rate=None):
    if sample_rate == None:
        sample_rate = 0.01
    if processedDataFolder == None:
        processedDataFolder = "datasets"
    if outputFileName == None:
        outputFileName = "au_informed"
    if relationName == None:
        relationName = "au_informed_accoustic"

    classes = sorted([f for f in os.listdir(ofFeaturesFolder)
                      if os.path.isdir(os.path.join(ofFeaturesFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    headerFlag = True
    header = []
    matrix = []
    tags = []
    tagIndices = []
    tagsFlag = False
    for className in classes:
        fileNames = sorted([f for f in os.listdir(os.path.join(ofFeaturesFolder, className))
                            if os.path.isfile(os.path.join(ofFeaturesFolder, className, f))
                            and not f.startswith('.') and f[-4:].lower() == ".csv"],
                           key=lambda f: f.lower())
        for name in fileNames:
            resultingVector = []
            with open(os.path.join(ofFeaturesFolder, className, name)) as visualFile:
                visualMatrix = list(csv.reader(visualFile))
            if not tagsFlag:
                firstRow = visualMatrix[0]
                for idx, item in enumerate(firstRow):
                    if "_r" in item:
                        tags.append(item.strip())
                        tagIndices.append(idx)
                tagsFlag = True

            aus = []
            for row in range(1, len(visualMatrix) - 1):
                maxVal = 0
                for tagIndex in tagIndices:
                    if float(visualMatrix[row][tagIndex]) > maxVal:
                        tag = visualMatrix[0][tagIndex].strip()
                        maxVal = float(visualMatrix[row][tagIndex])
                aus.append("%s;%s;%s" % (tag, visualMatrix[row][2].strip(), visualMatrix[row + 1][2].strip()))

            with open(os.path.join(accousticFolder, className, name)) as accousticFile:
                reader = csv.reader(accousticFile)
                accousticMatrix = []
                initialTime = -sample_rate
                endTime = 0
                for row in reader:
                    accousticMatrix.append([initialTime, endTime] + row)
                    initialTime += sample_rate
                    endTime += sample_rate
            for tag in tags:
                if headerFlag:
                    for feature in accousticMatrix[0][2:-1]:
                        header.append("%s_%s" % (tag.strip(), feature))
                listOfTimes = []
                for frame in aus:
                    elements = frame.split(";")
                    if elements[0].lower() == tag.lower():
                        listOfTimes.append([float(time) for time in elements[1:]])
                index = 0
                rows = []
                for window in listOfTimes:
                    for i in range(index + 1, len(accousticMatrix)):
                        if accousticMatrix[i][0] >= window[0]:
                            j = i
                            while window[1] > accousticMatrix[j][1]:
                                rows.append(accousticMatrix[j])
                                j += 1
                                if j >= len(accousticMatrix):
                                    break
                            index = j - 1
                            break
                if len(rows) > 0:
                    values = [[float(number) for number in row[2:]] for row in rows]
                else:
                    values = [[0.0] * len(accousticMatrix[0][2:-1])]
                array = np.array(values)
                for i in range(len(accousticMatrix[0][2:-1])):
                    column = array[:, i]
                    average = np.sum(column) / len(column)
                    if np.isinf(average):
                        resultingVector.append(np.sign(average))
                    elif np.isnan(average):
                        resultingVector.append(0)
                    else:
                        resultingVector.append(average)
            matrix.append(resultingVector + [className])
            headerFlag = False
    header = header + ["Class"]
    matrix = [header] + matrix
    am.create_arff(matrix, classes, processedDataFolder, outputFileName, relationName)
    print("AUs informed representation acquired.")


def syntax_informed_au(textsFolder, ofFeaturesFolder, processedDataFolder=None, outputFileName=None, relationName=None):
    if processedDataFolder == None:
        processedDataFolder = "datasets"
    if outputFileName == None:
        outputFileName = "syntax_informed_au"
    if relationName == None:
        relationName = "syntax_informed_au"

    tags = t.load_tags()
    classes = sorted([f for f in os.listdir(textsFolder)
                      if os.path.isdir(os.path.join(textsFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    headerFlag = True
    header = []
    matrix = []
    for className in classes:
        fileNames = sorted([f for f in os.listdir(os.path.join(textsFolder, className, "pos_timestamps"))
                            if os.path.isfile(os.path.join(textsFolder, className, "pos_timestamps", f))
                            and not f.startswith('.') and f[-4:].lower() == ".csv"],
                           key=lambda f: f.lower())
        for name in fileNames:
            resultingVector = []
            with open(os.path.join(textsFolder, className, "pos_timestamps", name)) as syntaxFile:
                line = syntaxFile.readline()
                posStamps = line.split(",")
            with open(os.path.join(ofFeaturesFolder, className, name)) as accousticFile:
                reader = list(csv.reader(accousticFile))
            visualMatrix = []
            auIndices = []
            row = []
            for idx, element in enumerate(reader[0]):
                if "timestamp" in element:
                    timeIndex = idx
                    row += ["ST", "FT"]
                if "_c" in element:
                    auIndices.append(idx)
                    row.append(element.strip())
            visualMatrix.append(row)
            for idx, row in enumerate(reader[1:-1]):
                visualRow = [float(row[timeIndex]), float(reader[idx + 2][timeIndex])]
                for index in auIndices:
                    visualRow.append(row[index])
                visualMatrix.append(visualRow)
            for tag in tags:
                if headerFlag:
                    for feature in visualMatrix[0][2:]:
                        header.append("%s_%s" % (tag, feature))
                listOfTimes = []
                for pos in posStamps:
                    elements = pos.split(";")
                    if elements[0].lower() == tag.lower():
                        listOfTimes.append([float(time) for time in elements[1:]])
                index = 0
                rows = []
                for window in listOfTimes:
                    for i in range(index + 1, len(visualMatrix)):
                        if visualMatrix[i][0] >= window[0]:
                            j = i
                            while window[1] > visualMatrix[j][1]:
                                rows.append(visualMatrix[j])
                                j += 1
                                if j >= len(visualMatrix):
                                    break
                            index = j - 1
                            break
                if len(rows) > 0:
                    values = [[float(number) for number in row[2:]] for row in rows]
                else:
                    values = [[0.0] * len(visualMatrix[0][2:])]
                array = np.array(values)
                for i in range(len(visualMatrix[0][2:])):
                    column = array[:, i]
                    average = np.sum(column) / len(column)
                    if np.isinf(average):
                        resultingVector.append(np.sign(average))
                    elif np.isnan(average):
                        resultingVector.append(0)
                    else:
                        resultingVector.append(average)
            matrix.append(resultingVector + [className])
            headerFlag = False
    header = header + ["Class"]
    matrix = [header] + matrix
    am.create_arff(matrix, classes, processedDataFolder, outputFileName, relationName)


def au_informed_liwc(ofFeaturesFolder, textsFolder, processedDataFolder=None, outputFileName=None, relationName=None,
                     lang=None):
    if processedDataFolder == None:
        processedDataFolder = "datasets"
    if outputFileName == None:
        outputFileName = "au_informed_liwc"
    if relationName == None:
        relationName = "au_informed_liwc"
    if lang == 'English':
        liwcDictionary = "LIWC2007_English131104.dic"
    elif lang == 'Spanish':
        liwcDictionary = "LIWC2007_Spanish.dic"
    else:
        liwcDictionary = "LIWC2007_English131104.dic"

    tags = []
    liwcDict = dictLib.Dictionary(os.path.join("text/Dictionaries/", liwcDictionary))
    classes = sorted([f for f in os.listdir(ofFeaturesFolder)
                      if os.path.isdir(os.path.join(ofFeaturesFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    headerFlag = True
    header = []
    matrix = []
    for className in classes:
        fileNames = sorted([f for f in os.listdir(os.path.join(ofFeaturesFolder, className))
                            if os.path.isfile(os.path.join(ofFeaturesFolder, className, f))
                            and not f.startswith('.') and f[-4:].lower() == ".csv"],
                           key=lambda f: f.lower())
        for name in fileNames:
            resultingVector = []
            with open(os.path.join(textsFolder, className, name.replace(".csv", "_timestamps.csv"))) as syntaxFile:
                line = syntaxFile.readline()
                wordStamps = [element.split(";") for element in line.split(",")]
            with open(os.path.join(ofFeaturesFolder, className, name)) as visualFile:
                reader = list(csv.reader(visualFile))
            visualMatrix = []
            auIndices = []
            row = []
            tags = []
            for idx, element in enumerate(reader[0]):
                if "timestamp" in element:
                    timeIndex = idx
                    row += ["ST", "FT"]
                if "_c" in element:
                    auIndices.append(idx)
                    row.append(element.strip())
                    tags.append(element.strip())
            visualMatrix.append(row)
            for idx, row in enumerate(reader[1:-1]):
                visualRow = [float(row[timeIndex]), float(reader[idx + 2][timeIndex])]
                for index in auIndices:
                    visualRow.append(row[index])
                visualMatrix.append(visualRow)

            for tag in tags:
                if headerFlag:
                    for feature in liwcDict.names:
                        header.append("%s_%s" % (tag, feature))
                listOfTimes = []
                auIndex = visualMatrix[0].index(tag)
                end = -1.0
                for idx, row in enumerate(visualMatrix[1:-1], 1):
                    if float(row[auIndex]) == 1.0:
                        begin = float(row[0])
                        if begin == end:
                            end = float(row[1])
                            listOfTimes[-1][1] = end
                        else:
                            end = float(row[1])
                            listOfTimes.append([begin, end])
                index = 0
                tokens = []
                for window in listOfTimes:
                    for i in range(index, len(wordStamps)):
                        if float(wordStamps[i][1]) >= window[0]:
                            j = i
                            while window[1] > float(wordStamps[j][2]):
                                tokens.append(wordStamps[j][0].lower())
                                j += 1
                                if j >= len(wordStamps):
                                    break
                            index = j
                            break
                values = dictLib.vectorize(tokens, liwcDict)
                resultingVector += values
            matrix.append(resultingVector + [className])
            headerFlag = False
    header = header + ["Class"]
    matrix = [header] + matrix
    am.create_arff(matrix, classes, processedDataFolder, outputFileName, relationName)


class BSSD:
    def __init__(self, booster, modality_folder=None, databases_folder=None, k_max=50, file_exceptions=[]):
        self.booster = booster
        self.k_max = k_max
        if databases_folder == None:
            databases_folder = os.getcwd()
        self.databases_folder = databases_folder
        self.train_data, self.labels, self.classes, self.attributes, self.dataset_names = \
            self.arffs_to_matrices(os.path.join(databases_folder, modality_folder), file_exceptions)
        self.modality = modality_folder
        self.boost_estimators = [[base.clone(self.booster), -1] for i in range(k_max)]
        self.model_weights = [[1.0 / len(self.labels) for i in range(len(self.labels))] for j in range(k_max)]
        self.alpha = [0 for i in range(k_max)]
        self.errors = [1.0 for i in range(k_max)]
        self.plots_path = ""
        self.set_plots_path()

    def arffs_to_matrices(self, folder, exceptions):

        modality_matrices = []
        modality_attributes = []
        view_names = []
        view_labels = []
        views = sorted([f for f in os.listdir(folder)
                        if os.path.isfile(os.path.join(folder, f)) and not f.startswith('.')
                        and f.endswith(".arff") and f not in exceptions], key=lambda f: f.lower())
        for view in views:
            matrix, labels, relation, attributes = am.arff_to_nparray(os.path.join(folder, view))
            modality_matrices.append(matrix)
            modality_attributes.append(tuple(attributes))
            view_names.append(relation)
            view_labels.append(labels)
        modality_matrices = tuple(modality_matrices)
        modality_attributes = tuple(modality_attributes)
        views = tuple(views)
        view_labels = tuple(view_labels)

        initial_labels = view_labels[0].tolist()
        for labels in view_labels:
            if labels.tolist() != initial_labels:
                print("There is a mismatch between the labels from each view.")
                raise RuntimeError
        classes = list(set(initial_labels))
        classes.sort()

        return modality_matrices, view_labels[0], tuple(classes), modality_attributes, views

    def check_cv(self, cv, labels):
        return verify_folds(cv, labels)

    def cross_val_predict(self, cv=10, plot_errors=False, plot_best_views=False, plot_distributions=False):

        labels = ["None" for i in range(len(self.labels))]
        folds = self.check_cv(cv, self.labels)
        num_folds = len(folds)

        now = time.time()
        print("Prediction started at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        # list_of_labels = Parallel(n_jobs=1) \
        list_of_labels = Parallel(n_jobs=min(len(folds), multiprocessing.cpu_count())) \
            (delayed(self.train_folds)(fold, pair, num_folds, plot_errors, plot_best_views, plot_distributions)
             for fold, pair in enumerate(folds))

        for fold, pair in enumerate(folds):
            for idx, value in enumerate(list_of_labels[fold]):
                position = pair[1][idx]
                labels[position] = value
        print("Prediction finished in", round(time.time() - now, 3), "sec at",
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        return np.array(labels)

    def cross_val_score(self, cv=10, plot_errors=False, plot_best_views=False, plot_distributions=False):

        labels = ["None" for i in range(len(self.labels))]
        folds = self.check_cv(cv, self.labels)
        num_folds = len(folds)

        now = time.time()
        print("Prediction started at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        # list_of_labels = Parallel(n_jobs=1) \
        list_of_labels = Parallel(n_jobs=min(len(folds), multiprocessing.cpu_count())) \
            (delayed(self.train_folds)(fold, pair, num_folds, plot_errors, plot_best_views, plot_distributions)
             for fold, pair in enumerate(folds))

        accuracy = []
        auc = []
        for fold, pair in enumerate(folds):
            real_labels = self.labels[pair[1]]
            predicted_labels = list_of_labels[fold]
            accuracy.append(metrics.accuracy_score(real_labels, predicted_labels))
            auc.append(
                metrics.roc_auc_score(preprocessing.label_binarize(real_labels, classes=list(reversed(self.classes))),
                                      preprocessing.label_binarize(predicted_labels,
                                                                   classes=list(reversed(self.classes)))))
            for idx, value in enumerate(list_of_labels[fold]):
                position = pair[1][idx]
                labels[position] = value
        print("Prediction finished in", round(time.time() - now, 3), "sec at",
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        return np.array(accuracy), np.array(auc), np.array(labels)

    def fit(self, plot_errors=False, plot_best_views=False, plot_distributions=False):

        print("Building model...")
        now = time.time()
        self.model_weights = self.train(self.train_data, self.labels, self.model_weights)
        print("Model built in %s sec" % (time.time() - now))
        if plot_errors:
            self.plot_errors()
        if plot_best_views:
            self.plot_best_views()
        if plot_distributions:
            self.plot_distributions()

    def get_plots_path(self):
        return self.plots_path

    def plot_best_views(self, destination=None, filename=None):
        if destination == None:
            destination = os.path.join(os.getcwd(), "bssd_plots", "best_views")
        if filename == None:
            filename = self.modality
        num_views = len(self.dataset_names)
        title = "Best views during training for %s modality" % (self.modality)
        data = [estimator[1] for estimator in self.boost_estimators]
        plots.s3db_best_views(data, num_views, title, destination, filename)

    def plot_distributions(self, destination=None):
        if destination == None:
            destination = os.path.join(os.getcwd(), "bssd_plots", "distributions")
        title = "Training Weights - %s modality" % (self.modality)
        data = self.model_weights
        views = self.dataset_names
        best_views = [estimator[1] for estimator in self.boost_estimators]
        folder = os.path.join(destination, self.modality)
        plots.s3db_distributions(data, views, best_views, title, folder)

    def plot_errors(self, destination=None, filename=None):
        if destination == None:
            destination = os.path.join(os.getcwd(), "bssd_plots", "errors")
        if filename == None:
            filename = self.modality
        title = "Errors during training for %s modality" % (self.modality)
        data = self.errors
        plots.s3db_errors(data, title, destination, filename)

    def predict(self, dataset, boost_estimators=None, alpha=None):

        if boost_estimators is None:
            boost_estimators = self.boost_estimators
        if alpha is None:
            alpha = self.alpha

        predicted_labels = []
        hyphotesis = [0 for i in range(len(dataset[0]))]
        for k in range(self.k_max):
            prediction = boost_estimators[k][0].predict(dataset[boost_estimators[k][1]])
            predicted_labels.append(
                preprocessing.label_binarize(prediction, neg_label=-1, classes=list(reversed(self.classes)))
            )
        for i in range(len(dataset[0])):
            sum = 0
            for k in range(self.k_max):
                sum += float(predicted_labels[k][i]) * alpha[k]
            if sum < 0:
                hyphotesis[i] = list(reversed(self.classes))[0]
            else:
                hyphotesis[i] = list(reversed(self.classes))[1]
        hyphotesis = np.array(hyphotesis)
        return hyphotesis

    def set_classes(self, classes):
        self.classes = tuple(classes)

    def set_plots_path(self, path=None):
        if path is not None:
            self.plots_path = path
        else:
            self.plots_path = os.path.join(os.getcwd(), "complementarity_analysis", "plots", "bssd_plots")

    def train(self, train_data, train_labels, model_weights, num_folds=0, cross_validating=False):

        if not cross_validating:
            boost_estimators = self.boost_estimators
            alpha = self.alpha
        else:
            boost_estimators = deepcopy(self.boost_estimators)
            alpha = deepcopy(self.alpha)
        train_view_data = [train_data, train_labels, []]
        for k in range(self.k_max):
            sample_weights = list(model_weights[k])
            train_view_data[-1] = sample_weights
            # errors = Parallel(n_jobs=1, prefer="threads") \
            errors = Parallel(n_jobs=min(multiprocessing.cpu_count()-num_folds, len(train_data)), prefer="threads")\
                (delayed(self.train_views)(view, train_view_data) for view in range(len(train_data)))
            min_error = 1.0
            for index, error in enumerate(errors):
                if error <= min_error:
                    min_error = error
                    best_view = index
            self.errors[k] = min_error
            boost_estimators[k][0] = deepcopy(self.booster)
            boost_estimators[k][0].fit(train_data[best_view], train_labels, sample_weights)
            boost_estimators[k][1] = best_view
            alpha[k] = 0.5 * math.log((1.0 - min_error) / min_error)
            sum = 0
            predicted_labels = boost_estimators[k][0].predict(train_data[best_view])
            if k + 1 < self.k_max:
                for sample in range(len(train_labels)):
                    if predicted_labels[sample] == train_labels[sample]:
                        result = model_weights[k][sample] * math.exp(-alpha[k])
                    else:
                        result = model_weights[k][sample] * math.exp(alpha[k])
                    sum += result
                    model_weights[k + 1][sample] = result
                for sample in range(len(train_labels)):
                    model_weights[k + 1][sample] = model_weights[k + 1][sample] / sum
        if not cross_validating:
            return model_weights
        else:
            return boost_estimators, alpha

    def train_folds(self, fold, pair, num_folds, plot_errors, plot_best_views, plot_distributions):

        now = time.time()
        train_labels = self.labels[pair[0]]
        train_data = []
        test_data = []
        for view in self.train_data:
            train_data.append(view[pair[0]])
            test_data.append(view[pair[1]])
        train_data = tuple(train_data)
        test_data = tuple(test_data)
        model_weights = [[1.0 / len(train_labels) for i in range(len(train_labels))] for j in range(self.k_max)]
        boost_estimators, alpha = self.train(train_data, train_labels, model_weights, num_folds, True)
        if plot_errors:
            self.plot_errors(destination=
                             os.path.join(self.get_plots_path(), "errors", "Fold%s" % (fold))
                             )
        if plot_best_views:
            self.plot_best_views(destination=
                                 os.path.join(self.get_plots_path(), "best_views", "Fold%s" % (fold))
                                 )
        if plot_distributions:
            self.plot_distributions(destination=
                                    os.path.join(self.get_plots_path(), "distributions", "Fold%s" % (fold))
                                    )
        predicted_test_labels = self.predict(test_data, boost_estimators, alpha)
        print("Model built for fold %s in %s sec" % (fold + 1, time.time() - now))
        return predicted_test_labels

    def train_views(self, view, train_view_data):

        weak_learner = deepcopy(self.booster)
        weak_learner.fit(train_view_data[0][view], train_view_data[1], train_view_data[2])
        predicted_labels = weak_learner.predict(train_view_data[0][view])
        current_error = 1.0 - metrics.accuracy_score(train_view_data[1], predicted_labels,
                                                     sample_weight=train_view_data[2])
        if current_error < 0.0001:
            current_error = 0.0001
        elif current_error > 0.9999:
            current_error = 0.9999
        return current_error


class S3DB(BSSD):
    def __init__(self, booster, stacker, modality_folders=None, databases_folder=None, k_max=50, file_exceptions=[]):
        self.booster = booster
        self.stacker = stacker
        self.k_max = k_max
        if databases_folder == None:
            databases_folder = os.getcwd()
        self.databases_folder = databases_folder
        if modality_folders == None:
            modality_folders = sorted([f for f in os.listdir(databases_folder)
                                       if os.path.isdir(os.path.join(databases_folder, f)) and not f.startswith('.')],
                                      key=lambda f: f.lower())
            self.train_data, self.labels, self.attributes, self.dataset_names = \
                self.arffs_to_matrices([os.path.join(databases_folder, modality) for modality in modality_folders],
                                       file_exceptions)
        else:
            if type(modality_folders) is not list:
                print("Third argument must be a list of strings")
                raise TypeError
            else:
                self.train_data, self.labels, self.classes, self.attributes, self.dataset_names = \
                    self.arffs_to_matrices([os.path.join(databases_folder, modality) for modality in modality_folders],
                                           file_exceptions)
        self.modalities = tuple(modality_folders)
        self.boost_estimators = [[[base.clone(self.booster), -1] for i in range(k_max)]
                                 for i in range(len(self.train_data))]
        self.model_weights = [[[1.0 / len(self.labels) for i in range(len(self.labels))]
                               for j in range(k_max)] for k in range(len(self.train_data))]
        self.alpha = [[0 for i in range(k_max)] for j in range(len(self.train_data))]
        self.errors = [[1.0 for i in range(k_max)] for j in range(len(self.train_data))]
        self.plots_path = ""
        self.set_plots_path()

    def arffs_to_matrices(self, folders, exceptions):
        matrices = []
        dataset_attributes = []
        dataset_names = []
        label_list = []

        for modality in folders:
            modality_matrices = []
            modality_attributes = []
            view_names = []
            view_labels = []
            views = sorted([f for f in os.listdir(modality)
                            if os.path.isfile(os.path.join(modality, f)) and not f.startswith('.')
                            and f.endswith(".arff") and f not in exceptions], key=lambda f: f.lower())
            for view in views:
                matrix, labels, relation, attributes = am.arff_to_nparray(os.path.join(modality, view))
                modality_matrices.append(matrix)
                modality_attributes.append(tuple(attributes))
                view_names.append(relation)
                view_labels.append(labels)
            matrices.append(tuple(modality_matrices))
            dataset_attributes.append(tuple(modality_attributes))
            dataset_names.append(tuple(views))
            label_list.append(tuple(view_labels))

        initial_labels = label_list[0][0].tolist()
        for modality_labels in label_list:
            for view_labels in modality_labels:
                if view_labels.tolist() != initial_labels:
                    print("There is a mismatch between the labels from each view.")
                    raise RuntimeError
        classes = list(set(initial_labels))
        classes.sort()

        return tuple(matrices), label_list[0][0], tuple(classes), tuple(dataset_attributes), tuple(dataset_names)

    # def cross_val_predict(self, cv=10, plot_errors=False, plot_best_views=False, plot_distributions=False):
    #
    #     labels = ["None" for i in range(len(self.labels))]
    #     folds = self.check_cv(cv, self.labels)
    #     num_folds = len(folds)
    #
    #     now = time.time()
    #     print("Prediction started at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    #     # list_of_labels = Parallel(n_jobs=1) \
    #     list_of_labels = Parallel(n_jobs=min(len(folds), multiprocessing.cpu_count())) \
    #         (delayed(self.train_folds)(fold, pair, num_folds, plot_errors, plot_best_views, plot_distributions)
    #          for fold, pair in enumerate(folds))
    #
    #     for fold, pair in enumerate(folds):
    #         for idx, value in enumerate(list_of_labels[fold]):
    #             position = pair[1][idx]
    #             labels[position] = value
    #     print("Prediction finished in", round(time.time() - now, 3), "sec at",
    #           datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    #     return np.array(labels)
    #
    # def cross_val_score(self, cv=10, plot_errors=False, plot_best_views=False, plot_distributions=False):
    #
    #     labels = ["None" for i in range(len(self.labels))]
    #     folds = self.check_cv(cv, self.labels)
    #     num_folds = len(folds)
    #
    #     now = time.time()
    #     print("Prediction started at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    #     # list_of_labels = Parallel(n_jobs=1) \
    #     list_of_labels = Parallel(n_jobs=min(len(folds), multiprocessing.cpu_count())) \
    #         (delayed(self.train_folds)(fold, pair, num_folds, plot_errors, plot_best_views, plot_distributions)
    #          for fold, pair in enumerate(folds))
    #
    #     accuracy = []
    #     auc = []
    #     for fold, pair in enumerate(folds):
    #         real_labels = self.labels[pair[1]]
    #         predicted_labels = list_of_labels[fold]
    #         accuracy.append(metrics.accuracy_score(real_labels, predicted_labels))
    #         auc.append(
    #             metrics.roc_auc_score(preprocessing.label_binarize(real_labels, classes=list(reversed(self.classes))),
    #                                   preprocessing.label_binarize(predicted_labels,
    #                                                                classes=list(reversed(self.classes)))))
    #         for idx, value in enumerate(list_of_labels[fold]):
    #                         position = pair[1][idx]
    #                         labels[position] = value
    #
    #     print("Prediction finished in", round(time.time() - now, 3), "sec at",
    #           datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    #     return accuracy, auc, np.array(labels)

    def fit(self, plot_errors=False, plot_best_views=False, plot_distributions=False):

        print("Building model...")
        now = time.time()
        self.model_weights = self.train(self.train_data, self.labels, self.model_weights)
        if plot_errors:
            self.plot_errors()
        if plot_best_views:
            self.plot_best_views()
        if plot_distributions:
            self.plot_distributions()
        stacking_dataset = self.get_stacking_dataset(self.train_data, self.labels)
        self.stacker.fit(stacking_dataset, self.labels)
        print("Model built in %s sec" % (time.time() - now))

    def get_stacking_dataset(self, dataset, labels, boost_estimators=None, alpha=None):

        if boost_estimators is None:
            boost_estimators = self.boost_estimators
        if alpha is None:
            alpha = self.alpha
        stacking_dataset = [[0 for i in range(len(dataset))] for j in range(len(labels))]
        for idx_modality, modality in enumerate(dataset):
            predicted_labels = []
            for k in range(self.k_max):
                prediction = boost_estimators[idx_modality][k][0].predict(
                    dataset[idx_modality][boost_estimators[idx_modality][k][1]])
                predicted_labels.append(
                    preprocessing.label_binarize(prediction, neg_label=-1, classes=list(reversed(self.classes))))
            for i in range(len(labels)):
                sum = 0
                for k in range(self.k_max):
                    sum += float(predicted_labels[k][i]) * alpha[idx_modality][k]
                stacking_dataset[i][idx_modality] = sum
        stacking_dataset = np.array(stacking_dataset)
        return stacking_dataset

    def plot_best_views(self, destination=None, filenames=[]):
        if destination == None:
            destination = os.path.join(os.getcwd(), "s3db_plots", "best_views")
        for idx, modality in enumerate(self.modalities):
            try:
                filename = filenames[idx]
            except:
                filename = modality
            num_views = len(self.dataset_names[idx])
            title = "Best views during training for %s modality" % (modality)
            data = [estimator[1] for estimator in self.boost_estimators[idx]]
            plots.s3db_best_views(data, num_views, title, destination, filename)

    def plot_distributions(self, destination=None):
        if destination == None:
            destination = os.path.join(os.getcwd(), "s3db_plots", "distributions")
        for idx, modality in enumerate(self.modalities):
            title = "Training Weights - %s modality" % (modality)
            data = self.model_weights[idx]
            views = self.dataset_names[idx]
            best_views = [estimator[1] for estimator in self.boost_estimators[idx]]
            folder = os.path.join(destination, modality)
            plots.s3db_distributions(data, views, best_views, title, folder)

    def plot_errors(self, destination=None, filenames=[]):
        if destination == None:
            destination = os.path.join(os.getcwd(), "s3db_plots", "errors")
        for idx, modality in enumerate(self.modalities):
            try:
                filename = filenames[idx]
            except:
                filename = modality
            title = "Errors during training for %s modality" % (modality)
            data = self.errors[idx]
            plots.s3db_errors(data, title, destination, filename)

    def set_plots_path(self, path=None):
        if path is not None:
            self.plots_path = path
        else:
            self.plots_path = os.path.join(os.getcwd(), "complementarity_analysis", "plots", "s3db_plots")

    def train(self, train_data, train_labels, model_weights, num_folds=0, cross_validating=False):

        if not cross_validating:
            boost_estimators = self.boost_estimators
            alpha = self.alpha
        else:
            boost_estimators = deepcopy(self.boost_estimators)
            alpha = deepcopy(self.alpha)

        for idx_modality, modality in enumerate(train_data):
            train_view_data = [train_data[idx_modality], train_labels, []]
            for k in range(self.k_max):
                sample_weights = list(model_weights[idx_modality][k])
                train_view_data[-1] = sample_weights
                # errors = Parallel(n_jobs=1, prefer="threads") \
                errors = Parallel(n_jobs=min(multiprocessing.cpu_count() - num_folds, len(train_data)),
                                  prefer="threads") \
                    (delayed(self.train_views)(view, train_view_data) for view in range(len(train_data[idx_modality])))
                min_error = 1.0
                for index, error in enumerate(errors):
                    if error <= min_error:
                        min_error = error
                        best_view = index
                self.errors[idx_modality][k] = min_error
                boost_estimators[idx_modality][k][0] = deepcopy(self.booster)
                boost_estimators[idx_modality][k][0].fit(train_data[idx_modality][best_view], train_labels,
                                                         sample_weights)
                boost_estimators[idx_modality][k][1] = best_view
                alpha[idx_modality][k] = 0.5 * math.log((1.0 - min_error) / min_error)
                sum = 0
                predicted_labels = boost_estimators[idx_modality][k][0].predict(train_data[idx_modality][best_view])
                if k + 1 < self.k_max:
                    for sample in range(len(train_labels)):
                        if predicted_labels[sample] == train_labels[sample]:
                            result = model_weights[idx_modality][k][sample] * math.exp(
                                -alpha[idx_modality][k])
                        else:
                            result = model_weights[idx_modality][k][sample] * math.exp(
                                alpha[idx_modality][k])
                        sum += result
                        model_weights[idx_modality][k + 1][sample] = result
                    for sample in range(len(train_labels)):
                        model_weights[idx_modality][k + 1][sample] = \
                            model_weights[idx_modality][k + 1][sample] / sum
        if not cross_validating:
            return model_weights
        else:
            return boost_estimators, alpha

    def train_folds(self, fold, pair, num_folds, plot_errors, plot_best_views, plot_distributions):

        now = time.time()
        train_labels = self.labels[pair[0]]
        test_labels = self.labels[pair[1]]
        train_data = []
        test_data = []
        for modality in self.train_data:
            modality_train_data = []
            modality_test_data = []
            for view in modality:
                modality_train_data.append(view[pair[0]])
                modality_test_data.append(view[pair[1]])
            train_data.append(tuple(modality_train_data))
            test_data.append(tuple(modality_test_data))
        train_data = tuple(train_data)
        test_data = tuple(test_data)
        model_weights = [[[1.0 / len(train_labels) for i in range(len(train_labels))]
                          for j in range(self.k_max)] for k in range(len(train_data))]
        boost_estimators, alpha = self.train(train_data, train_labels, model_weights, num_folds, True)
        if plot_errors:
            self.plot_errors(destination=
                             os.path.join(self.get_plots_path(), "errors", "Fold%s" % (fold))
                             )
        if plot_best_views:
            self.plot_best_views(destination=
                                 os.path.join(self.get_plots_path(), "best_views", "Fold%s" % (fold))
                                 )
        if plot_distributions:
            self.plot_distributions(destination=
                                    os.path.join(self.get_plots_path(), "distributions", "Fold%s" % (fold))
                                    )
        train_stacking_dataset = self.get_stacking_dataset(train_data, train_labels, boost_estimators, alpha)
        stacker = deepcopy(self.stacker)
        stacker.fit(train_stacking_dataset, train_labels)
        test_stacking_dataset = self.get_stacking_dataset(test_data, test_labels, boost_estimators, alpha)
        predicted_test_labels = stacker.predict(test_stacking_dataset)
        print("Model built for fold %s in %s sec" % (fold + 1, time.time() - now))
        return predicted_test_labels


class BSSD2(BSSD):

    def set_plots_path(self, path=None):
        if path is not None:
            self.plots_path = path
        else:
            self.plots_path = os.path.join(os.getcwd(), "complementarity_analysis", "plots", "bssd_cv_plots")

    def sub_cross_val_predict(self, sub_pair, view, train_view_data):
        weak_learner = deepcopy(self.booster)
        sub_train_labels = train_view_data[1][sub_pair[0]]
        sub_train_data = train_view_data[0][view][sub_pair[0]]
        sub_test_data = train_view_data[0][view][sub_pair[1]]
        sub_model_weights = train_view_data[2][sub_pair[0]].tolist()
        weak_learner.fit(sub_train_data, sub_train_labels, sub_model_weights)
        predicted_test_labels = weak_learner.predict(sub_test_data)
        return predicted_test_labels

    def train_views(self, view, train_view_data):

        predicted_labels = ["None" for i in range(len(train_view_data[1]))]
        sub_folds = self.check_cv(9, train_view_data[1])
        train_view_data[2] = np.array(train_view_data[2])
        # list_of_labels = Parallel(n_jobs=1)(
        #     delayed(self.sub_cross_val_predict)(sub_pair, view, train_view_data) for sub_pair in sub_folds)
        list_of_labels = model.cross_val_predict(deepcopy(self.booster), train_view_data[0][view], train_view_data[1], cv=sub_folds)
        sub_accs = []
        for sub_fold, sub_pair in enumerate(sub_folds):
            # for idx, value in enumerate(list_of_labels[sub_fold]):
            #     position = sub_pair[1][idx]
            #     predicted_labels[position] = value
            sub_true_labels = train_view_data[1][sub_pair[1]]
            sub_predicted_labels = list_of_labels[sub_pair[1]]
            sub_weights = train_view_data[2][sub_pair[1]]
            sub_weights = sub_weights / sum(sub_weights)
            sub_acc = metrics.accuracy_score(sub_true_labels, sub_predicted_labels, sample_weight=sub_weights)
            sub_accs.append(sub_acc)

        current_error = 1- np.array(sub_accs).mean()
        # current_error = 1.0 - metrics.accuracy_score(train_view_data[1], predicted_labels,
        #                                              sample_weight=train_view_data[2])
        if current_error < 0.0001:
            current_error = 0.0001
        elif current_error > 0.9999:
            current_error = 0.9999
        return current_error


class BSSD2_2(BSSD):

    def cross_val_predict(self, cv, plot_errors=False, plot_best_views=False, plot_distributions=False):

        labels = ["None" for i in range(len(self.labels))]
        folds = self.check_cv(cv[0], self.labels)
        num_folds = len(folds)

        now = time.time()
        print("Prediction started at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        # list_of_labels = Parallel(n_jobs=1) \
        list_of_labels = Parallel(n_jobs=min(len(folds), multiprocessing.cpu_count())) \
            (delayed(self.train_folds)(fold, pair, num_folds, cv[1][fold][0], plot_errors, plot_best_views,
                                       plot_distributions)
             for fold, pair in enumerate(folds))

        for fold, pair in enumerate(folds):
            for idx, value in enumerate(list_of_labels[fold]):
                position = pair[1][idx]
                labels[position] = value
        print("Prediction finished in", round(time.time() - now, 3), "sec at",
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        return np.array(labels)

    def cross_val_score(self, cv, plot_errors=False, plot_best_views=False, plot_distributions=False):

        labels = ["None" for i in range(len(self.labels))]
        folds = self.check_cv(cv[0], self.labels)
        num_folds = len(folds)

        now = time.time()
        print("Prediction started at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        # list_of_labels = Parallel(n_jobs=1) \
        list_of_labels = Parallel(n_jobs=min(len(folds), multiprocessing.cpu_count())) \
            (delayed(self.train_folds)(fold, pair, num_folds, cv[1][fold][0], plot_errors, plot_best_views,
                                       plot_distributions)
             for fold, pair in enumerate(folds))

        accuracy = []
        auc = []
        for fold, pair in enumerate(folds):
            real_labels = self.labels[pair[1]]
            predicted_labels = list_of_labels[fold]
            accuracy.append(metrics.accuracy_score(real_labels, predicted_labels))
            auc.append(
                metrics.roc_auc_score(preprocessing.label_binarize(real_labels, classes=list(reversed(self.classes))),
                                      preprocessing.label_binarize(predicted_labels,
                                                                   classes=list(reversed(self.classes)))))
            for idx, value in enumerate(list_of_labels[fold]):
                position = pair[1][idx]
                labels[position] = value
        print("Prediction finished in", round(time.time() - now, 3), "sec at",
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        return np.array(accuracy), np.array(auc), np.array(labels)

    def get_nested_cross_iterable(self, custom_dict, folds=9, seed=None, return_custom_dict=False):
        return sa.get_nested_cross_iterable(custom_dict, folds, seed, return_custom_dict)

    def set_plots_path(self, path=None):
        if path is not None:
            self.plots_path = path
        else:
            self.plots_path = os.path.join(os.getcwd(), "complementarity_analysis", "plots", "bssd_cv_plots")

    def sub_cross_val_predict(self, sub_pair, view, train_view_data):
        weak_learner = deepcopy(self.booster)
        sub_train_labels = train_view_data[1][sub_pair[0]]
        sub_train_data = train_view_data[0][view][sub_pair[0]]
        sub_test_data = train_view_data[0][view][sub_pair[1]]
        sub_model_weights = train_view_data[2][sub_pair[0]].tolist()
        weak_learner.fit(sub_train_data, sub_train_labels, sub_model_weights)
        predicted_test_labels = weak_learner.predict(sub_test_data)
        return predicted_test_labels

    def train(self, train_data, train_labels, custom_dict, model_weights, num_folds=0, cross_validating=False):

        if not cross_validating:
            boost_estimators = self.boost_estimators
            alpha = self.alpha
        else:
            boost_estimators = deepcopy(self.boost_estimators)
            alpha = deepcopy(self.alpha)
        train_view_data = [train_data, train_labels, []]
        for k in range(self.k_max):
            sample_weights = list(model_weights[k])
            train_view_data[-1] = sample_weights
            # errors = Parallel(n_jobs=1, prefer="threads") \
            errors = Parallel(n_jobs=min(multiprocessing.cpu_count() - num_folds, len(train_data)), prefer="threads") \
                (delayed(self.train_views)(view, train_view_data, custom_dict) for view in range(len(train_data)))
            min_error = 1.0
            for index, error in enumerate(errors):
                if error <= min_error:
                    min_error = error
                    best_view = index
            self.errors[k] = min_error
            boost_estimators[k][0] = deepcopy(self.booster)
            boost_estimators[k][0].fit(train_data[best_view], train_labels, sample_weights)
            boost_estimators[k][1] = best_view
            alpha[k] = 0.5 * math.log((1.0 - min_error) / min_error)
            sum = 0
            predicted_labels = boost_estimators[k][0].predict(train_data[best_view])
            if k + 1 < self.k_max:
                for sample in range(len(train_labels)):
                    if predicted_labels[sample] == train_labels[sample]:
                        result = model_weights[k][sample] * math.exp(-alpha[k])
                    else:
                        result = model_weights[k][sample] * math.exp(alpha[k])
                    sum += result
                    model_weights[k + 1][sample] = result
                for sample in range(len(train_labels)):
                    model_weights[k + 1][sample] = model_weights[k + 1][sample] / sum
        if not cross_validating:
            return model_weights
        else:
            return boost_estimators, alpha

    def train_folds(self, fold, pair, num_folds, custom_dict, plot_errors, plot_best_views, plot_distributions):

        now = time.time()
        train_labels = self.labels[pair[0]]
        train_data = []
        test_data = []
        for view in self.train_data:
            train_data.append(view[pair[0]])
            test_data.append(view[pair[1]])
        train_data = tuple(train_data)
        test_data = tuple(test_data)
        model_weights = [[1.0 / len(train_labels) for i in range(len(train_labels))] for j in range(self.k_max)]
        boost_estimators, alpha = self.train(train_data, train_labels, custom_dict, model_weights, num_folds, True)
        if plot_errors:
            self.plot_errors(destination=
                             os.path.join(self.get_plots_path(), "errors", "Fold%s" % (fold))
                             )
        if plot_best_views:
            self.plot_best_views(destination=
                                 os.path.join(self.get_plots_path(), "best_views", "Fold%s" % (fold))
                                 )
        if plot_distributions:
            self.plot_distributions(destination=
                                    os.path.join(self.get_plots_path(), "distributions", "Fold%s" % (fold))
                                    )
        predicted_test_labels = self.predict(test_data, boost_estimators, alpha)
        print("Model built for fold %s in %s sec" % (fold + 1, time.time() - now))
        return predicted_test_labels

    def train_views(self, view, train_view_data, custom_dict):

        predicted_labels = ["None" for i in range(len(train_view_data[1]))]
        sub_folds = self.get_nested_cross_iterable(custom_dict)
        sub_folds = self.check_cv(sub_folds, train_view_data[1])
        train_view_data[2] = np.array(train_view_data[2])
        # list_of_labels = Parallel(n_jobs=1)(
        #     delayed(self.sub_cross_val_predict)(sub_pair, view, train_view_data) for sub_pair in sub_folds)
        list_of_labels = model.cross_val_predict(deepcopy(self.booster), train_view_data[0][view], train_view_data[1], cv=sub_folds)
        sub_accs = []
        for sub_fold, sub_pair in enumerate(sub_folds):
            # for idx, value in enumerate(list_of_labels[sub_fold]):
            #     position = sub_pair[1][idx]
            #     predicted_labels[position] = value
            sub_true_labels = train_view_data[1][sub_pair[1]]
            sub_predicted_labels = list_of_labels[sub_pair[1]]
            sub_weights = train_view_data[2][sub_pair[1]]
            sub_weights = sub_weights/sum(sub_weights)
            sub_acc = metrics.accuracy_score(sub_true_labels, sub_predicted_labels, sample_weight=sub_weights)
            sub_accs.append(sub_acc)

        current_error = 1- np.array(sub_accs).mean()
        # current_error = 1.0 - metrics.accuracy_score(train_view_data[1], predicted_labels,
        #                                              sample_weight=train_view_data[2])
        if current_error < 0.0001:
            current_error = 0.0001
        elif current_error > 0.9999:
            current_error = 0.9999
        return current_error


class S3DB2(S3DB):

    def train(self, train_data, train_labels, model_weights, num_folds=0, cross_validating=False):

        sys.stdout.write("\r%d%%" % (0))
        sys.stdout.flush()
        for idx_modality, modality in enumerate(train_data):
            for k in range(self.k_max):
                sample_weights = np.array(list(model_weights[idx_modality][k]))
                min_error = 1.0
                for idx_view, view in enumerate(modality):
                    predicted_labels = ["None" for i in range(len(self.labels))]
                    sub_folds = self.check_cv(9, train_labels)
                    for sub_fold, sub_pair in enumerate(sub_folds):
                        sub_train_labels = self.labels[sub_pair[0]]
                        sub_train_data = view[sub_pair[0]]
                        sub_test_data = view[sub_pair[1]]
                        sub_model_weights = sample_weights[sub_pair[0]].tolist()
                        self.boost_estimators[idx_modality][k][0].fit(sub_train_data, sub_train_labels,
                                                                      sub_model_weights)
                        predicted_test_labels = self.boost_estimators[idx_modality][k][0].predict(sub_test_data)
                        for idx, value in enumerate(predicted_test_labels):
                            position = sub_pair[1][idx]
                            predicted_labels[position] = value
                    current_error = 1.0 - metrics.accuracy_score(train_labels, predicted_labels,
                                                                 sample_weight=sample_weights)
                    if current_error < 0.0001:
                        current_error = 0.0001
                    elif current_error > 0.9999:
                        current_error = 0.9999
                    if current_error <= min_error:
                        best_estimator = tuple([deepcopy(self.boost_estimators[idx_modality][k][0]), idx_view])
                        min_error = current_error
                    sys.stdout.write("\r%d%%" % (
                        round((k + 1 + (idx_modality + idx_view / len(modality)) * self.k_max) * 100.0 / (
                                self.k_max * len(self.modalities)))))
                    sys.stdout.flush()
                self.errors[idx_modality][k] = min_error
                self.boost_estimators[idx_modality][k][0] = deepcopy(best_estimator[0])
                self.boost_estimators[idx_modality][k][1] = best_estimator[1]
                self.alpha[idx_modality][k] = 0.5 * math.log((1.0 - min_error) / min_error)
                sum = 0
                predicted_labels = best_estimator[0].predict(train_data[idx_modality][best_estimator[1]])
                if k + 1 < self.k_max:
                    for sample in range(len(train_labels)):
                        if predicted_labels[sample] == train_labels[sample]:
                            result = model_weights[idx_modality][k][sample] * math.exp(
                                -self.alpha[idx_modality][k])
                        else:
                            result = model_weights[idx_modality][k][sample] * math.exp(
                                self.alpha[idx_modality][k])
                        sum += result
                        model_weights[idx_modality][k + 1][sample] = result
                    for sample in range(len(train_labels)):
                        model_weights[idx_modality][k + 1][sample] = \
                            model_weights[idx_modality][k + 1][sample] / sum
                sys.stdout.write(
                    "\r%d%%" % (
                        round((k + 1 + idx_modality * self.k_max) * 100.0 / (self.k_max * len(self.modalities)))))
                sys.stdout.flush()
        return model_weights

    def set_plots_path(self, path=None):
        if path is not None:
            self.plots_path = path
        else:
            self.plots_path = os.path.join(os.getcwd(), "complementarity_analysis", "plots", "s3db_cv_plots")


class S4DB(BSSD):

    def __init__(self, booster, stacker, modality_folder=None, databases_folder=None, k_max=50, file_exceptions=[]):
        self.booster = booster
        self.stacker = stacker
        self.k_max = k_max
        if databases_folder == None:
            databases_folder = os.getcwd()
        self.databases_folder = databases_folder
        self.train_data, self.labels, self.classes, self.attributes, self.dataset_names = \
            self.arffs_to_matrices(os.path.join(databases_folder, modality_folder), file_exceptions)
        self.modality = modality_folder
        self.boost_estimators = [[base.clone(self.booster), -1] for i in range(k_max)]
        self.model_weights = [[1.0 / len(self.labels) for i in range(len(self.labels))] for j in range(k_max)]
        self.alpha = [0 for i in range(k_max)]
        self.errors = [1.0 for i in range(k_max)]
        self.plots_path = ""
        self.set_plots_path()

    def fit(self, plot_errors=False, plot_best_views=False, plot_distributions=False):

        print("Building model...")
        now = time.time()
        self.model_weights = self.train(self.train_data, self.labels, self.model_weights)
        stacking_dataset = self.get_stacking_dataset(self.train_data)
        self.stacker.fit(stacking_dataset, self.labels)
        print("Model built in %s sec" % (time.time() - now))

        if plot_errors:
            self.plot_errors()
        if plot_best_views:
            self.plot_best_views()
        if plot_distributions:
            self.plot_distributions()

    def get_stacking_dataset(self, dataset, boost_estimators=None, alpha=None):

        if boost_estimators is None:
            boost_estimators = self.boost_estimators
        if alpha is None:
            alpha = self.alpha

        predicted_labels = []
        for k in range(self.k_max):
            prediction = boost_estimators[k][0].predict(
                dataset[boost_estimators[k][1]]
            )
            prediction = prediction.reshape(-1, 1)
            predicted_labels.append(
                preprocessing.label_binarize(prediction, neg_label=-1, classes=list(reversed(self.classes))))
        predicted_labels = tuple(predicted_labels)
        stacking_dataset = np.column_stack(predicted_labels)
        return stacking_dataset

    def predict(self, dataset):

        stacking_dataset = self.get_stacking_dataset(dataset)
        hypothesis = self.stacker.predict(stacking_dataset)
        return hypothesis

    def set_plots_path(self, path=None):
        if path is not None:
            self.plots_path = path
        else:
            self.plots_path = os.path.join(os.getcwd(), "complementarity_analysis", "plots", "stacking_bssd_plots")

    def train_folds(self, fold, pair, num_folds, plot_errors, plot_best_views, plot_distributions):

        now = time.time()
        train_labels = self.labels[pair[0]]
        test_labels = self.labels[pair[1]]
        train_data = []
        test_data = []
        for view in self.train_data:
            train_data.append(view[pair[0]])
            test_data.append(view[pair[1]])
        train_data = tuple(train_data)
        test_data = tuple(test_data)
        model_weights = [[1.0 / len(train_labels) for i in range(len(train_labels))] for j in range(self.k_max)]
        boost_estimators, alpha = self.train(train_data, train_labels, model_weights, num_folds, True)
        if plot_errors:
            self.plot_errors(destination=
                             os.path.join(self.get_plots_path(), "errors", "Fold%s" % (fold))
                             )
        if plot_best_views:
            self.plot_best_views(destination=
                                 os.path.join(self.get_plots_path(), "best_views", "Fold%s" % (fold))
                                 )
        if plot_distributions:
            self.plot_distributions(destination=
                                    os.path.join(self.get_plots_path(), "distributions", "Fold%s" % (fold))
                                    )
        train_stacking_dataset = self.get_stacking_dataset(train_data, boost_estimators, alpha)
        stacker = deepcopy(self.stacker)
        stacker.fit(train_stacking_dataset, train_labels)
        test_stacking_dataset = self.get_stacking_dataset(test_data, boost_estimators, alpha)
        predicted_test_labels = stacker.predict(test_stacking_dataset)
        print("Model built for fold %s in %s sec" % (fold + 1, time.time() - now))
        return predicted_test_labels
