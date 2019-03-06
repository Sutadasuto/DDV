from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import nltk
from os import listdir
from os.path import isfile, join
import os


def extract_bot(databaseFolder, terms, targetFileFolder, outputFileName, corpusThreshold=None, saveArff=None):

    if corpusThreshold == None:
        corpusThreshold = 1
    if saveArff == None:
        saveArff = False

    if not os.path.exists(targetFileFolder):
        os.makedirs(targetFileFolder)

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    vectorizer = CountVectorizer(token_pattern=r'\S+')
    data_corpus = []
    labels = []
    with open(os.path.join(targetFileFolder, "%s.txt"%(outputFileName)), "w+") as pf:
        for className in classes:

            fileNames = sorted([f for f in listdir(join(databaseFolder, className, terms))
                                if isfile(join(databaseFolder, className, terms, f))
                                and not f.startswith('.') and f[-4:].lower() == ".txt"],
                               key=lambda f: f.lower())
            for fileName in fileNames:
                pf.write(fileName + "\n")
                classFolder = join(databaseFolder, className, terms)
                with open(join(classFolder, fileName)) as f:
                    text = " ".join(line.strip() for line in f)
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
            vectorizer = CountVectorizer(token_pattern=r'\S+')
            X = vectorizer.fit_transform(data_corpus)
        elif corpusThreshold >= 0 and corpusThreshold <= 1:
            vectorizer = CountVectorizer(token_pattern=r'\S+', min_df=corpusThreshold)
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
        if saveArff:
            with open(join(targetFileFolder, outputFileName + ".arff"), 'w+') as result:
                result.write('@relation ' + terms + '\n\n')
                for name in header:
                    result.write('@attribute %s_term numeric\n' % (name))
                result.write('@attribute Class {%s}\n\n' % (",".join(classes)))
                result.write('@data\n')
                for row in matrix:
                    result.write(','.join(['{:.4f}'.format(x) for x in row[:-1]]) + ',' + row[-1] + '\n')

    return header, matrix


def extract_bot_from_vocabulary(databaseFolder, terms, trainingVocabulary, targetFileFolder, outputFileName, saveArff=None):

    if saveArff == None:
        saveArff = False
    if not os.path.exists(targetFileFolder):
        os.makedirs(targetFileFolder)

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    vectorizer = CountVectorizer(vocabulary = trainingVocabulary)
    data_corpus = []
    labels = []
    with open(os.path.join(targetFileFolder, "%s.txt"%(outputFileName)), "w+") as pf:
        for className in classes:

            fileNames = sorted([f for f in listdir(join(databaseFolder, className, terms))
                                if isfile(join(join(databaseFolder, className, terms), f))
                                and not f.startswith('.') and f[-4:].lower() == ".txt"],
                               key=lambda f: f.lower())
            for fileName in fileNames:
                pf.write(fileName + "\n")
                with open(join(databaseFolder, className, fileName)) as f:
                    text = " ".join(line.strip() for line in f)
                labels.append(className)
                data_corpus.append(text.lower())

    if len(data_corpus) > 0:
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
        if saveArff:
            with open(join(targetFileFolder, "%s.arff"%(outputFileName)), 'w+') as result:
                result.write('@relation ' + outputFileName + '\n\n')
                for name in header:
                    result.write('@attribute %s_term numeric\n' % (name))
                result.write('@attribute Class {%s}\n\n' % (",".join(classes)))
                result.write('@data\n')
                for row in matrix:
                    result.write(','.join(['{:.4f}'.format(x) for x in row[:-1]]) + ',' + row[-1] + '\n')
        return header, matrix


def extract_bow(databaseFolder, targetFileFolder, outputFileName, corpusThreshold=None, saveArff=None):

    if corpusThreshold == None:
        corpusThreshold = 1
    if saveArff == None:
        saveArff = False

    if not os.path.exists(targetFileFolder):
        os.makedirs(targetFileFolder)

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    vectorizer = CountVectorizer(token_pattern=r'\S+')
    data_corpus = []
    labels = []
    analyzedFiles = []

    for className in classes:
        fileNames = sorted([f for f in os.listdir(os.path.join(databaseFolder, className))
                            if os.path.isfile(os.path.join(databaseFolder, className, f))
                            and not f.startswith('.') and f[-4:].lower() == ".txt"],
                           key=lambda f: f.lower())
        analyzedFiles += ["%s,%s" % (file, className) for file in fileNames]
        for fileName in fileNames:
            with open(join(databaseFolder, className, fileName)) as f:
                text =  " ".join(line.strip() for line in f)
            try:
                tokens = nltk.word_tokenize(text)
            except:
                tokens = nltk.word_tokenize(text.decode('utf-8'))
            text = " ".join(tokens)
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
            vectorizer = CountVectorizer(token_pattern=r'\S+')
            X = vectorizer.fit_transform(data_corpus)
        elif corpusThreshold >= 0 and corpusThreshold <= 1:
            vectorizer = CountVectorizer(token_pattern=r'\S+', min_df=corpusThreshold)
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
        if saveArff:
            with open(join(targetFileFolder, outputFileName + ".arff"), 'w+') as result:
                result.write('@relation bow\n\n')
                for name in header:
                    result.write('@attribute "%s_word" numeric\n' % (name))
                result.write('@attribute Class {%s}\n\n' % (",".join(classes)))
                result.write('@data\n')
                for row in matrix:
                    result.write(','.join(['{:.4f}'.format(x) for x in row[:-1]]) + ',' + row[-1] + '\n')
    with open(os.path.join(targetFileFolder, "%s.txt" % (outputFileName)), "w+") as files:
        files.write("\n".join(analyzedFiles))
    return header, matrix


def extract_bow_from_vocabulary(databaseFolder, trainingVocabulary, targetFileFolder, outputFileName, saveArff=None):

    if saveArff == None:
        saveArff = False

    if not os.path.exists(targetFileFolder):
        os.makedirs(targetFileFolder)

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    vectorizer = CountVectorizer(vocabulary=trainingVocabulary)
    data_corpus = []
    labels = []
    analyzedFiles = []

    for className in classes:
        fileNames = sorted([f for f in os.listdir(os.path.join(databaseFolder, className))
                            if os.path.isfile(os.path.join(databaseFolder, className, f))
                            and not f.startswith('.') and f[-4:].lower() == ".txt"],
                           key=lambda f: f.lower())
        analyzedFiles += fileNames
        for fileName in fileNames:
            with open(join(databaseFolder, className, fileName)) as f:
                text =  " ".join(line.strip() for line in f)
            try:
                tokens = nltk.word_tokenize(text)
            except:
                tokens = nltk.word_tokenize(text.decode('utf-8'))
            text = " ".join(tokens)
            labels.append(className)
            data_corpus.append(text.lower())

    if len(data_corpus) > 0:
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
        if saveArff:
            if saveArff:
                with open(join(targetFileFolder, outputFileName + ".arff"), 'w+') as result:
                    result.write('@relation bow_with_prev_vocabulary' + '\n\n')
                    for name in header:
                        result.write('@attribute "%s_word" numeric\n' % (name))
                    result.write('@attribute Class {%s}\n\n' % (",".join(classes)))
                    result.write('@data\n')
                    for row in matrix:
                        result.write(','.join(['{:.4f}'.format(x) for x in row[:-1]]) + ',' + row[-1] + '\n')
    with open(os.path.join(targetFileFolder, "%s.txt" % (outputFileName)), "w+") as files:
        files.write("\n".join(analyzedFiles))
    return header, matrix


def get_bow(databaseFolder, targetFileFolder, corpusThreshold=None):

    if corpusThreshold == None:
        corpusThreshold = 1

    if not os.path.exists(targetFileFolder):
        os.makedirs(targetFileFolder)

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    vectorizer = CountVectorizer(token_pattern=r'\S+')
    data_corpus = []
    labels = []
    for className in classes:
        fileNames = sorted([f for f in os.listdir(os.path.join(databaseFolder, className))
                            if os.path.isfile(os.path.join(databaseFolder, className, f))
                            and not f.startswith('.') and f[-4:].lower() == ".txt"],
                           key=lambda f: f.lower())

        for fileName in fileNames:
            with open(join(databaseFolder, className, fileName)) as f:
                text =  " ".join(line.strip() for line in f)
            labels.append(className)
            data_corpus.append(text.lower())

    if len(data_corpus) > 0:
        X = vectorizer.fit_transform(data_corpus)
        if corpusThreshold > 1:
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

    return vectorizer.get_feature_names(), matrix


def get_sentence_wise_bow(sentences, vocabulary):

    vectorizer = CountVectorizer(vocabulary=vocabulary)
    X = vectorizer.fit_transform(sentences)
    data = X.toarray()
    data = preprocessing.normalize(data, norm='l1')
    matrix = []
    for row in data:
        vector = row.tolist()
        matrix.append(vector)
    return matrix