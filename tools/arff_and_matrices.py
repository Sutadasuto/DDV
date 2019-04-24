import os
import csv
import numpy as np
import arff #liac-arff
from tools import plots


def arff_to_nparray(filePath):

    with open(filePath) as f:
        dataDictionary = arff.load(f)
        arffData = np.array(dataDictionary['data'])
        X = arffData[:, :-1]
        y = arffData[:, -1]
        matrix = X.astype(float)
        attributes = [str(e[0]) for e in dataDictionary["attributes"][:-1]]
        relation = str(dataDictionary["relation"])
        labels = y.astype(str)

    return matrix, labels, relation, attributes


def arffs_to_matrix(inputFolder=None, fileNames=None, exceptions=[]):

    if inputFolder == None:
        inputFolder = "datasets"
    if len(exceptions) == 0:
        exceptions = ["early_fusion.arff", "syntax_informed.arff"]
    if fileNames == None:
        fileNames = sorted([f for f in os.listdir(inputFolder)
                            if os.path.isfile(os.path.join(inputFolder, f))
                            and not f.startswith('.') and f[-5:].lower() == ".arff"
                            and not f in exceptions],
                           key=lambda f: f.lower())

    modalities = "-".join(fileNames)
    modalities = modalities.replace(".arff", "")

    fileLocations = [os.path.join(inputFolder, fileName) for fileName in fileNames]

    attributeNames = []
    attributeValues = []
    labels = []

    with open(fileLocations[0]) as arff:
        lines = arff.readlines()
        names = []
        values = []
        for line in lines:
            if line.startswith("@attribute "):
                names.append(line.split("@attribute ")[1])
            elif not line.startswith("@") and not line.startswith("\n"):
                lineValues = line.split(",")
                labels.append(lineValues[-1])
                values.append([float(value) for value in lineValues[:-1]])

    classes = names.pop()
    attributeNames.append(names)
    attributeValues.append(values)


    for fileLocation in fileLocations[1:]:
        with open(fileLocation) as arff:
            lines = arff.readlines()
            names = []
            values = []
            for line in lines:
                if line.startswith("@attribute "):
                    names.append(line.split("@attribute ")[1])
                elif not line.startswith("@") and not line.startswith("\n"):
                    lineValues = line.split(",")
                    values.append([float(value) for value in lineValues[:-1]])

        attributeNames.append(names[:-1])
        attributeValues.append(values)

    header = []
    for set in attributeNames:
        for attribute in set:
            header.append(attribute.split(" numeric")[0])
    header.append("Class")

    data = []
    for rowNum in range(len(labels)):
        row = []
        for set in attributeValues:
            for value in set[rowNum]:
                row.append(value)
        row.append(labels[rowNum].split("\n")[0])
        data.append(row)

    matrix = [header] + data

    return [matrix, classes.split("{")[1].split("}")[0].split(","), modalities]


def arffs_to_matrices(processedDataFolder=None):

    if processedDataFolder == None:
        processedDataFolder = "datasets"

    arffFiles = sorted([f for f in os.listdir(processedDataFolder)
                        if os.path.isfile(os.path.join(processedDataFolder, f))
                        and not f.startswith('.') and f[-5:].lower() == ".arff"],
                       key=lambda f: f.lower())

    relations = ["" for i in range(len(arffFiles))]
    listsOfAttributes = [[] for i in range(len(arffFiles))]
    matrices = [np.array([]) for i in range(len(arffFiles))]

    for idx, arffFile in enumerate(arffFiles):
        with open(os.path.join("datasets", arffFile)) as f:
            dataDictionary = arff.load(f)
            arffData = np.array(dataDictionary['data'])
            X = arffData[:, :-1]
            y = arffData[:, -1]
            matrices[idx] = X.astype(float)
            attributes = [str(e[0]) for e in dataDictionary["attributes"][:-1]]
            listsOfAttributes[idx] = attributes
            relations[idx] = str(dataDictionary["relation"])
    labels = y.astype(str)

    return relations, matrices, labels, listsOfAttributes


def complementarity_comparison(results, header, destiny_folder, name="complementarity_comparison", plot_title="Generic Title"):

    with open(os.path.join(destiny_folder, "%s.xlsx" % (name)), "w+") as csvfile:
        writer = csv.writer(csvfile)
        firstColumn = np.array([[""] + header]).transpose()
        data = np.array(results).transpose()
        statistics = [["Average", "", "Std. Dev.", ""], ["CFD", "MPA", "CFD", "MPA"]]
        for row in data[2:]:
            dataCFD = []
            for i in range(0,len(row),2):
                dataCFD.append(float(row[i]))
            dataMPA = []
            for i in range(1, len(row), 2):
                dataMPA.append(float(row[i]))
            statistics.append([round(np.average(dataCFD),3),
                               round(np.average(dataMPA),1),
                               round(np.std(dataCFD),3),
                               round(np.std(dataMPA),1)])
        statistics = np.array(statistics).astype(str)
        complementarityMatrix = np.column_stack((firstColumn, data, statistics)).tolist()
        plots.plot_complementarity_matrix(complementarityMatrix, plot_title,
                                          destiny_folder)
        writer.writerows(complementarityMatrix)


def create_arff(matrix, classes, targetFileFolder=None, fileName=None, relationName=None):

    if targetFileFolder == None:
        targetFileFolder = "datasets"
    if fileName == None:
        fileName = "multimodal"
    if relationName == None:
        relationName = fileName

    generateARFF(targetFileFolder,fileName,relationName,matrix,classes)


def generateARFF(targetFileFolder, fileName, relationName, matrix, classes):
    if not os.path.exists(targetFileFolder):
        os.makedirs(targetFileFolder)

    header = matrix[0]
    with open(os.path.join(targetFileFolder, fileName + ".arff"), 'w+') as result:
        result.write('@relation ' + relationName + '\n\n')
        for name in header[:-1]:
            result.write('@attribute ' + name + ' numeric\n')

        string = '@attribute ' + header[-1] + ' {'
        for label in classes:
            string += label + ","
        string = string[:-1] + "}"
        result.write(string)
        result.write('\n\n@data\n')
        for row in matrix[1:]:
            if len(row) > 1:
                result.write(','.join(['{:.4f}'.format(float(x)) for x in row[:-1]]) + ',' + row[-1] + '\n')


def get_metrics_from_matrix(matrix):

    vectors = [[] for i in range(len(matrix[0]-2))]

    for i in range(len(matrix[0] - 2)):
        for j in range(1,len(matrix)):
            vectors[i].append([matrix[j][1], matrix[j][i]])

    classes = []
    for instance in vectors[0]:
        if instance[0] not in classes:
            classes.append(instance[0])

    for vector in vectors:
        count = [0]


def matrices_comparison(results, first_column, destiny_folder, name="clf_comparison",
                        plot_title="Generic Title", plot_subtitles=None, category_end="all_"):

    with open(os.path.join(destiny_folder, "%s.xlsx" % (name)), "w+") as csvfile:
        writer = csv.writer(csvfile)
        data = np.column_stack(tuple(results))
        statistics = [["Average","","Std. Dev.",""],["Acc","AUC","Acc","AUC"]]
        for row in data[2:]:
            dataAcc = []
            for i in range(0,len(row),2):
                dataAcc.append(float(row[i]))
            dataAuc = []
            for i in range(1, len(row), 2):
                dataAuc.append(float(row[i]))
            statistics.append([round(np.average(dataAcc),1),
                               round(np.average(dataAuc),3),
                               round(np.std(dataAcc),1),
                               round(np.std(dataAuc),3)])
        statistics = np.array(statistics).astype(str)
        comparisonMatrix = np.column_stack((first_column, data, statistics)).tolist()
        plots.plot_classifiers_matrix(comparisonMatrix, plot_title,
                                      destiny_folder, subtitles=plot_subtitles, stopKey=category_end)
        writer.writerows(comparisonMatrix)


def separate_single_attributes(wekaWrapper, targetFile, classifierList=None, folds=None, targetFileFolder=None, fileName=None):

    if classifierList == None:
        classifierList = ["bayes.NaiveBayes", "functions.SGD", "functions.Logistic", "functions.MultilayerPerceptron",
                          "trees.J48", "trees.RandomForest"]
    if folds == None:
        folds = 10

    if targetFileFolder == None:
        targetFileFolder = "single_attribute_evaluations"

    [matrix, classes] = arffs_to_matrix([targetFile])

    for i in range(len(matrix[0])-1):
        subMatrix = []
        for row in matrix:
            subMatrix.append([row[i], row[-1]])
        create_arff(subMatrix, classes, targetFileFolder, subMatrix[0][0], subMatrix[0][0])

    wekaWrapper.classify_cross_validated(targetFileFolder, classifierList, folds, targetFileFolder)
    print(".csv results per classifier stored in '" + targetFileFolder + "'.")