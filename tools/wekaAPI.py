# Python 2.7
# python-weka-wrapper

import weka.core.jvm as jvm
import os
import csv
import copy
from tools.wekaExperiments import *

class Weka():

    def __init__(self):
        jvm.start()


    def jvmStop(self):
        jvm.stop()

    def classify_subject_cv(self, trainingFolder, testFolder, testList, classifier):

        classifierClass = "weka.classifiers."
        classifier = classifierClass + classifier

        testedInstances = []
        with open(testList) as testFiles:
            lines = testFiles.readlines()
            for line in lines:
                testedInstances.append(line.split(".")[0])

        datasetTrainingNames = sorted([f for f in os.listdir(trainingFolder)
                                       if os.path.isfile(os.path.join(trainingFolder, f))
                                           and not f.startswith('.') and f[-5:].lower() == ".arff"],
                                          key=lambda f: f.lower())
        datasetTestNames = sorted([f for f in os.listdir(testFolder)
                                       if os.path.isfile(os.path.join(testFolder, f))
                                           and not f.startswith('.') and f[-5:].lower() == ".arff"],
                                          key=lambda f: f.lower())

        if not datasetTestNames == datasetTrainingNames:
            print("There is mismatch between training and test sets")
            print("Training sets:")
            print (",".join(datasetTrainingNames))
            print("Test sets:")
            print (",".join(datasetTestNames))
            raise EnvironmentError

        multimodalMatrix = [[] for i in range(len(testedInstances) + 1)]
        startFlag = True
        for name in datasetTrainingNames:
            print("Dataset: " + name)
            trainingSet = os.path.join(trainingFolder, name)
            validationSet = os.path.join(testFolder, name)
            e = Experiment()
            [header, matrix] = e.train_and_separate_validation(trainingSet, validationSet, testedInstances, classifier)
            if startFlag:
                multimodalMatrix[0]+=[header[0][0], header[1][3], header[0][1].split("/")[-1].split(".")[0]]
                for i in range(1,len(testedInstances) + 1):
                    multimodalMatrix[i]+=[matrix[i-1][0], matrix[i-1][3], matrix[i-1][4]]
                startFlag=False
            else:
                multimodalMatrix[0]+=[header[0][1].split("/")[-1].split(".")[0]]
                for i in range(1,len(testedInstances) + 1):
                    multimodalMatrix[i]+=[matrix[i-1][4]]
        headerFlag = True
        for row in multimodalMatrix:
            if headerFlag:
                row.append("Guess Ratio")
                headerFlag = False
            else:
                guesses = row[2:]
                row.append(round(float(sum(guesses) / len(guesses)), 2))
        return multimodalMatrix




    def classify_cross_validated(self, datasetsFolder, classifierList, folds, outputFolder):

        classifierClass = "weka.classifiers."
        for idx, classifier in enumerate(classifierList):
            classifierList[idx] = classifierClass + classifier

        cases = sorted([f for f in os.listdir(datasetsFolder)
                        if os.path.isdir(os.path.join(datasetsFolder, f)) and not f.startswith('.')],
                       key=lambda f: f.lower())
        if len(cases) == 0:
            cases = [""]
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        for classifier in classifierList:

            print("Clasificador: " + classifier + "\n")
            with open(outputFolder + "/" + classifier + ".csv", "w+") as results:
                matrix = []

                for case in cases:
                    print("Experimento: " + case + ".\n\n")

                    trainingSets = sorted([f for f in os.listdir(datasetsFolder + "/" + case) if os.path.isfile(os.path.join(datasetsFolder + "/" + case, f))
                                           and not f.startswith('.') and f[-5:].lower() == ".arff"],
                                          key=lambda f: f.lower())

                    for trainingSet in trainingSets:

                        print("Atributos: " + trainingSet[:-5])
                        trainingFile = datasetsFolder + "/" + case + "/" + trainingSet
                        matrix.append([trainingSet[:-5]])
                        e = Experiment()
                        e.runCV(trainingFile, classifier, folds)
                        print(e.header)
                        print(e.values)
                        matrix.append([case] + e.values)
                        print("\n")

                featureResults = []
                for features in range(len(trainingSets)):
                    featureResult = []
                    featureResult.append(copy.deepcopy(matrix[2 * features]))
                    featureResult.append(["Experiment"] + e.header)
                    for set in range(2 * features + 1, len(cases) * 2 * len(trainingSets), 2 * len(trainingSets)):
                        featureResult.append(matrix[set])
                    featureResults.append(featureResult)

                wr = csv.writer(results)
                for result in featureResults:
                    wr.writerows(result)
                    wr.writerow([])


    def output_classifier_predictions(self, databases, processedFiles, classifierList, outputFolder=None):

        if outputFolder == None:
            outputFolder = "modality_performance"
        classifierClass = "weka.classifiers."
        for idx, classifier in enumerate(classifierList):
            classifierList[idx] = classifierClass + classifier

        cases = databases
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        matrix = [["File"],[""]]
        with open(processedFiles) as pf:
            lines = pf.readlines()
            for line in lines:
                matrix.append([line.replace("\n", "")])

        for classifier in classifierList:

            currentMatrix = copy.deepcopy(matrix)
            print("Clasificador: " + classifier + "\n")
            with open(outputFolder + "/" + classifier + ".csv", "w+") as results:

                for case in cases:
                    print("Experimento: " + case + ".\n\n")
                    e = Experiment()
                    [predictions, guess, head, realLabels] = e.train_and_predict_instances(case, classifier)
                    currentMatrix[0] += [case.split("/")[-1]] + [""]*(len(head)-1)
                    currentMatrix[1] += head
                    for i in range(len(realLabels)):
                        currentMatrix[i+2] += (predictions[i] + [guess[i]])
                currentMatrix[0] += []
                currentMatrix[1] += ["Correct Label"]
                for i in range(len(realLabels)):
                    currentMatrix[i+2] += [realLabels[i]]

                wr = csv.writer(results)
                wr.writerows(currentMatrix)