from text.lib.dictionary import *
from os import listdir
from os.path import isfile, join
import os


def get_liwc_analysis(databaseFolder, targetFileFolder, outputFileName, liwcDictionary):

    if not os.path.exists(targetFileFolder):
        os.makedirs(targetFileFolder)

    dictPath = join("text/Dictionaries/", liwcDictionary)
    dict = Dictionary(dictPath)

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    with open(os.path.join(targetFileFolder, "liwc.txt"), "w+") as pf:
        with open(join(targetFileFolder, outputFileName + ".arff"), 'w+') as result:
            result.write('@relation ' + outputFileName + '\n\n')
            header = dict.names
            for name in header:
                result.write('@attribute %s_liwc numeric\n'%(name))
            result.write('@attribute Class {%s}\n\n'%(",".join(classes)))
            result.write('@data\n')
            analyzedFiles = []
            for className in classes:
                experimentFolder = join(databaseFolder, className)
                fileNames = sorted([f for f in listdir(experimentFolder)
                                    if isfile(join(experimentFolder, f))
                                    and not f.startswith('.') and f[-4:].lower() == ".txt"],
                                   key=lambda f: f.lower())

                for fileName in fileNames:
                    analyzedFiles.append("%s,%s" % (fileName, className))
                    pf.write(fileName + "\n")
                    file = open(join(experimentFolder, fileName))
                    tokens = tokenize(file)
                    if len(tokens) == 0:
                        a=0
                    vector = vectorize(tokens, dict)
                    line = ",".join(['{:.4f}'.format(x) for x in vector]) + "," + className
                    result.write(line + "\n")
    with open(join(targetFileFolder, outputFileName + ".txt"), 'w+') as files:
        files.write("\n".join(analyzedFiles))
    print("LIWC vector representation acquired.")