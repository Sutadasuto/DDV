from os import listdir
from os.path import isfile, join
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import os


def extract_bag_of_pos_ngrams(n, databaseFolder, targetFileFolder, outputFileName, corpusThreshold=None):

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    vectorizer = CountVectorizer(token_pattern=r'[^\n]+')
    data_corpus = []
    labels = []
    analyzedFiles = []

    for className in classes:

        fileNames = sorted([f for f in listdir(join(databaseFolder, className, "pos"))
                            if isfile(join(join(databaseFolder, className, "pos"), f))
                            and not f.startswith('.') and f[-4:].lower() == ".txt"],
                           key=lambda f: f.lower())

        if not os.path.exists(targetFileFolder):
            os.makedirs(targetFileFolder)
        for fileName in fileNames:
            analyzedFiles.append("%s,%s" % (fileName, className))
            with open(join(databaseFolder, className, "pos", fileName)) as f:
                text = " ".join(line.strip() for line in f)
            text = ngrams(text.split(), n)
            string = []
            for grams in text:
                string.append(" ".join(list(grams)))
            text = "\n".join(string)
            labels.append(className)
            data_corpus.append(text.lower())

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
            vectorizer = CountVectorizer(token_pattern=r'[^\n]+')
            X = vectorizer.fit_transform(data_corpus)
        elif corpusThreshold >= 0 and corpusThreshold <= 1:
            vectorizer = CountVectorizer(token_pattern=r'[^\n]+', min_df=corpusThreshold)
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
                result.write('@attribute "%s_pos%sgram" numeric\n' % (name, n))
            result.write('@attribute Class {%s}\n\n' % (",".join(classes)))
            result.write('@data\n')
            for row in matrix:
                result.write(','.join(['{:.4f}'.format(x) for x in row[:-1]]) + ',' + row[-1] + '\n')

    with open(os.path.join(targetFileFolder, outputFileName + ".txt"), 'w+') as file:
        file.write("\n".join(analyzedFiles))

    return header, matrix