from os import listdir
from os.path import isfile, join
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import os


def extract_char_ngrams(n, databaseFolder):

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    for className in classes:
        fileNames = sorted([f for f in listdir(join(databaseFolder, className)) if isfile(join(databaseFolder, className, f))
                            and not f.startswith('.') and f[-4:].lower() == ".txt"],
                           key=lambda f: f.lower())
        targetFileFolder = join(databaseFolder, className, "char%sgrams"%(n))
        if not os.path.exists(targetFileFolder):
            os.makedirs(targetFileFolder)

        for file in fileNames:
            ngramsList = []
            with open(join(databaseFolder, className, file)) as corpus:
                for line in corpus:
                    sequence = line.lower().strip().decode('utf-8')
                    lineNgrams = ngrams(sequence, n)
                    ngramsList += lineNgrams

            with open(join(targetFileFolder, file), "w+") as targetFile:
                string = "\n".join(["".join(list(gram)) for gram in ngramsList])
                targetFile.write(string.encode('utf-8'))


def extract_bag_of_char_ngrams(n, databaseFolder, targetFileFolder, outputFileName, corpusThreshold=None):

    if corpusThreshold == None:
        corpusThreshold = 1

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    vectorizer = CountVectorizer(token_pattern=r'[^\n]{%s,%s}' % (n, n))
    data_corpus = []
    labels = []
    analyzedFiles = []

    for className in classes:
        fileNames = sorted([f for f in listdir(join(databaseFolder, className)) if isfile(join(databaseFolder, className, f))
                            and not f.startswith('.') and f[-4:].lower() == ".txt"],
                           key=lambda f: f.lower())
        if not os.path.exists(targetFileFolder):
            os.makedirs(targetFileFolder)

        for file in fileNames:
            analyzedFiles.append("%s,%s" % (file, className))
            ngramsList = []
            with open(join(databaseFolder, className, file)) as corpus:
                for line in corpus:
                    sequence = line.lower().strip()
                    lineNgrams = ngrams(sequence, n)
                    ngramsList += lineNgrams

            string = "\n".join(["".join(list(gram)) for gram in ngramsList])
            data_corpus.append(string)
            labels.append(className)

    if len(data_corpus) > 0:
        if corpusThreshold > 1:
            X = vectorizer.fit_transform(data_corpus)
            sum = X.sum(axis=0)
            indices = []
            for i in range(sum.size):
                if sum[0, i] > corpusThreshold:
                    indices.append(i)
            words = vectorizer.get_feature_names()
            vocabulary = []
            for index in indices:
                vocabulary.append(words[index])
            vectorizer = CountVectorizer(vocabulary=vocabulary)
            X = vectorizer.fit_transform(data_corpus)
        elif corpusThreshold == 1:
            vectorizer = CountVectorizer(token_pattern=r'[^\n]{%s,%s}' % (n, n))
            X = vectorizer.fit_transform(data_corpus)
        elif corpusThreshold >= 0 and corpusThreshold <= 1:
            vectorizer = CountVectorizer(token_pattern=r'[^\n]{%s,%s}' % (n, n), min_df=corpusThreshold)
            X = vectorizer.fit_transform(data_corpus)
        data = X.toarray()
        data = preprocessing.normalize(data, norm='l1')
        header = vectorizer.get_feature_names()
        index = 0
        matrix = []
        for row in data:
            vector = row.tolist()
            vector.append(labels[index])
            matrix.append(vector)
            index += 1
        with open(join(targetFileFolder, outputFileName + ".arff"), 'w+') as result:
            result.write('@relation %s\n\n' %(outputFileName))
            for name in header:
                result.write('@attribute "%s_char%sgram" numeric\n' % (name, n))
            result.write('@attribute Class {%s}\n\n' % (",".join(classes)))
            result.write('@data\n')
            for row in matrix:
                result.write(','.join(['{:.4f}'.format(x) for x in row[:-1]]) + ',' + row[-1] + '\n')

    with open(os.path.join(targetFileFolder, outputFileName + ".txt"), 'w+') as file:
        file.write("\n".join(analyzedFiles))

    return header, matrix