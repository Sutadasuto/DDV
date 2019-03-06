from text.lib import preprocess
from text.lib import liwc
from text.lib import charNgrams
from text.lib import ngrams
from text.lib import bot
from tools import config
import tools.arff_and_matrices as am
import os
import shutil
import csv
import nltk
import pandas
import scipy.stats
import numpy as np
import codecs
import subprocess
import csv

def dependency_distance(conll_df):
    """ Computes dependency distance for dependency tree. Based off of:
    Pakhomov, Serguei, et al. "Computerized assessment of syntactic complexity
    in Alzheimers disease: a case study of Iris Murdochs writing."
    Behavior research methods 43.1 (2011): 136-144."""
    ID = np.array([int(x) for x in conll_df['ID']])
    HEAD = np.array([int(x) for x in conll_df['HEAD']])
    diff = abs(ID - HEAD)
    total_distance = np.sum(diff)
    return total_distance


def extract_bag_of_char_ngrams(n, databaseFolder, targetFileFolder=None, outputFileName=None, corpusThreshold=None):

    if targetFileFolder == None:
        targetFileFolder = "datasets/textual"
    if outputFileName == None:
        outputFileName = "char%sgrams"%(n)

    header, matrix = charNgrams.extract_bag_of_char_ngrams(n, databaseFolder, targetFileFolder, outputFileName, corpusThreshold)
    print(str(n) + "-grams of characters acquired.")
    return header, matrix


def extract_bag_of_pos_ngrams(n, databaseFolder, targetFileFolder=None, outputFileName=None, corpusThreshold=None):

    if targetFileFolder == None:
        targetFileFolder = "datasets/textual"
    if outputFileName == None:
        outputFileName = "pos%sgrams"%(n)

    header, matrix = ngrams.extract_bag_of_pos_ngrams(n, databaseFolder, targetFileFolder, outputFileName, corpusThreshold)
    print(str(n) + "-grams of POS tags acquired.")
    return header, matrix


def get_bag_of_terms(databaseFolder, terms, targetFileFolder=None, outputFileName=None, corpusThreshold=None, saveArff=None):

    if targetFileFolder == None:
        targetFileFolder = "datasets"
    if outputFileName == None:
        outputFileName = "bag_of_%s"%(terms)

    nltk.download('punkt')
    vocabulary = bot.extract_bot(databaseFolder, terms, targetFileFolder, outputFileName, corpusThreshold, saveArff)

    print("Bag of " + terms + " representation acquired.")
    return vocabulary


def get_bot_from_vocabulary(databaseFolder, terms, trainingVocabulary,
                            targetFileFolder=None, outputFileName=None):

    if targetFileFolder == None:
        targetFileFolder = os.path.join(databaseFolder, terms + "_arff")
    if outputFileName == None:
        outputFileName = "bag_of_%s"%(terms)

    nltk.download('punkt')
    vocabularyUsed = bot.extract_bot_from_vocabulary(databaseFolder, terms, trainingVocabulary, targetFileFolder,
                                                     outputFileName)
    print("Bag of " + terms + " with custom vocabulary representation acquired.")


def get_bow(databaseFolder, targetFileFolder=None, outputFileName=None, corpusThreshold=None, saveArff=None):

    if targetFileFolder == None:
        targetFileFolder = "datasets/textual"
    if outputFileName == None:
        outputFileName = "bow"

    nltk.download('punkt')
    vocabulary = bot.extract_bow(databaseFolder, targetFileFolder, outputFileName, corpusThreshold, saveArff)

    print("Bag of words representation acquired.")
    return vocabulary


def get_features_per_category(databaseFolder, lang=None, n=None, targetFileFolder=None, corpusThreshold=None):

    if lang == None:
        lang = 'English'
    if n == None:
        n = [3]
    if targetFileFolder == None:
        targetFileFolder = "datasets/textual"
    if corpusThreshold == None:
        corpusThreshold = 0.10
    if lang == 'English':
        liwcDictionary = "LIWC2007_English131104.dic"
    elif lang == 'Spanish':
        liwcDictionary = "LIWC2007_Spanish.dic"

    liwc.get_liwc_analysis(databaseFolder, targetFileFolder, "liwc", liwcDictionary)
    vocabulary, bag = get_bow(databaseFolder, targetFileFolder, corpusThreshold=corpusThreshold, saveArff=True)

    syntax_header = [["word_count", "avg_wordlen", "levels", "distance", "univ_tag"]]
    syntax_header[0].append("Class")
    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    analyzedFiles = []
    matrix = []
    for className in classes:
        if not os.path.exists(os.path.join(databaseFolder, className, "pos")):
            os.makedirs(os.path.join(databaseFolder, className, "pos"))
        fileNames = sorted([f for f in os.listdir(os.path.join(databaseFolder, className))
                            if os.path.isfile(os.path.join(databaseFolder, className, f))
                            and not f.startswith('.') and f[-4:].lower() == ".txt"],
                           key=lambda f: f.lower())
        analyzedFiles += ["%s,%s" % (file, className) for file in fileNames]
        for fileName in fileNames:
            with open(os.path.join(databaseFolder, className, fileName)) as f:
                text = " ".join(line.strip() for line in f)
            conll_file = os.path.join(databaseFolder, className, "pos", fileName.replace(".txt", "_conll.csv"))
            if not os.path.exists(conll_file):
                conll = pos_tree_tag_sentence(text, lang)
                with open(os.path.join(databaseFolder, className, "pos", fileName.replace(".txt", "_conll.csv")),
                          "w+") as treeFile:
                    treeFile.write(conll)
            else:
                with open(os.path.join(databaseFolder, className, "pos", fileName.replace(".txt", "_conll.csv"))) as treeFile:
                    conll = treeFile.read()
            if len(conll) > 0:
                conll_lines = conll.strip().split('\n')
                conll_table = [line.split('\t') for line in conll_lines]
                df = pandas.DataFrame(conll_table,
                                      columns=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS',
                                               'MISC'])
                pos_feats = tag_count(df)  # pos tag count
                #print set(df['UPOS'].values)
                tags = list(set(df['UPOS'].values))
                with open(os.path.join(databaseFolder, className, "pos", fileName), "w+") as posFile:
                    posFile.write(" ".join(df['UPOS'].values))
                univ_tag = []
                for tag in tags:
                    if tag in load_tags():
                        univ_tag.append(tag)
                univ_tag = len(univ_tag)  # unique number of pos tags
                distance = dependency_distance(df)  # dependency distance
                heads = df['HEAD'].values
                levels = len(set(heads))  # tree depth
                words = preprocess.tokenize_sentence(text, lang)
                word_count = len(words)
                avg_wordlen = np.sum([len(w) for w in words]) / word_count # average word length
            else:
                with open(os.path.join(databaseFolder, className, "pos", fileName), "w+") as posFile:
                    posFile.write("")
                word_count = 0
                avg_wordlen = 0
                levels = 0
                distance = 0
                univ_tag = 0
            matrix.append([word_count, avg_wordlen, levels, distance, univ_tag, className])

    matrix = syntax_header + matrix
    am.generateARFF(targetFileFolder, "syntax_features", "syntax_features", matrix, classes)
    print("Syntax features acquired.")
    with open(os.path.join(targetFileFolder, "syntax_features.txt"), "w+") as files:
        files.write("\n".join(analyzedFiles))

    for value in n:
        vocabulary, bag = extract_bag_of_char_ngrams(value, databaseFolder, targetFileFolder, corpusThreshold=corpusThreshold)
        vocabulary, bag = extract_bag_of_pos_ngrams(value, databaseFolder, targetFileFolder, corpusThreshold=corpusThreshold)


def get_liwc_arff(databaseFolder, targetFileFolder=None, outputFileName=None, liwcDictionary=None):

    if liwcDictionary == None:
        liwcDictionary= "myLIWC.dic"
    if targetFileFolder == None:
        targetFileFolder = "datasets"
    if outputFileName == None:
        outputFileName = "LIWC_analysis"

    liwc.get_liwc_analysis(databaseFolder, targetFileFolder, outputFileName, liwcDictionary)
    print("LIWC vectors acquired.")


def get_statistics(databaseFolder, processedDataFolder=None, outputFileName=None, relationName=None):

    if processedDataFolder == None:
        processedDataFolder = "datasets"
    if outputFileName== None:
        outputFileName = "ling_features"
    if relationName == None:
        relationName = "ling_feature_statistics"

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    stats_names = ['max', 'min', 'mean', 'median', 'std', 'var', 'kurt', 'skew', 'percentile25', 'percentile50',
                   'percentile75']

    startFlag = True
    analyzedFiles = []
    for className in classes:
        files = sorted([f for f in os.listdir(os.path.join(databaseFolder, className))
                      if os.path.isfile(os.path.join(databaseFolder, className, f)) and not f.startswith('.')
                        and f[-4:].lower() == ".csv"], key=lambda f: f.lower())
        analyzedFiles += ["%s,%s" % (file, className) for file in files]
        for feat_file in files:
            mm_feats = []
            mm_names = []
            df = pandas.read_csv(os.path.join(databaseFolder, className, feat_file), header='infer')
            feature_names = df.columns.values
            for feat in feature_names:
                # Feature vector
                vals = df[feat].values
                # Run statistics
                maximum = np.nanmax(vals)
                minimum = np.nanmin(vals)
                mean = np.nanmean(vals)
                median = np.nanmedian(vals)
                std = np.nanstd(vals)
                var = np.nanvar(vals)
                kurt = scipy.stats.kurtosis(vals)
                skew = scipy.stats.skew(vals)
                percentile25 = np.nanpercentile(vals, 25)
                percentile50 = np.nanpercentile(vals, 50)
                percentile75 = np.nanpercentile(vals, 75)
                names = [feat.strip() + "_" + stat for stat in stats_names]
                feats = [maximum, minimum, mean, median, std, var, kurt, skew, percentile25, percentile50, percentile75]
                if startFlag:
                    for n in names:
                        mm_names.append(n)
                for f in feats:
                    mm_feats.append(f)
            if startFlag:
                matrix = [mm_names + ["Class"]]
                startFlag = False
            matrix.append(mm_feats + [className])
    am.create_arff(matrix,classes,processedDataFolder,outputFileName,relationName)
    with open(os.path.join(processedDataFolder, "ling_features.txt"), "w+") as files:
        files.write("\n".join(analyzedFiles))


def get_timestamped_pos_tags(databaseFolder, lang=None):

    if lang == None:
        lang = 'English'

    owd = os.getcwd()
    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    os.chdir(os.path.join(config.get_syntaxnet_folder(), "research", "syntaxnet"))

    for className in classes:
        fileNames = sorted([f for f in os.listdir(os.path.join(databaseFolder, className))
                            if os.path.isfile(os.path.join(databaseFolder, className, f))
                            and not f.startswith('.') and f[-4:].lower() == ".csv"],
                           key=lambda f: f.lower())
        if not os.path.exists(os.path.join(databaseFolder, className, "pos_timestamps")):
            os.makedirs(os.path.join(databaseFolder, className, "pos_timestamps"))
        for name in fileNames:
            if lang == 'Spanish':
                stamps = codecs.open(os.path.join(databaseFolder, className, name), 'r', 'utf-8').read()
                string = " ".join([line.split(";")[0] for line in stamps.strip().split(",")])
                command = 'echo "%s" | syntaxnet/models/parsey_universal/parse.sh syntaxnet/models/Spanish'%(string)
                string = string.replace(".", "").replace(",", "")
                output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
            elif lang == 'English':
                with open(os.path.join(databaseFolder, className, name)) as file:
                    stamps = file.readline()
                string = " ".join([line.split(";")[0].replace("%","") for line in stamps.strip().split(",")])
                string = string.replace(".", "").replace(",", "")
                command = 'echo "%s" | syntaxnet/demo.sh'%(string)
                output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
            posList = output.decode().strip()
            if len(posList) > 0:
                posList = posList.split("\n")
                stampsList = []
                contractions = ["wanna", "gotta", "gonna", "kinda"]
                for line in stamps.strip().split(","):
                    if len(line.split("'")) == 2:
                        stampsList.append(line.split(";")[1:])
                        stampsList.append(line.split(";")[1:])
                    elif line.split(";")[0].lower() in contractions:
                        stampsList.append(line.split(";")[1:])
                        stampsList.append(line.split(";")[1:])
                    else:
                        stampsList.append(line.split(";")[1:])
                string = ""
                for word in range(len(posList)):
                    conllLine = posList[word]
                    string += (conllLine.split("\t")[3] + ";")
                    string += (";".join(stampsList[word]) + ",")
                string = string[:-1]
            else:
                string = ""
            with open(os.path.join(databaseFolder, className, "pos_timestamps", name.replace("_timestamps", "")), "w+") as file:
                file.write(string)
    os.chdir(owd)
    print("POS tagging done.")


def load_tags():
    # Load universal POS tag set - http://universaldependencies.org/u/pos/all.html
    tags = "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".strip().split()
    return tags


def pos_tree_tag_sentence(sentence, lang=None):

    if lang == None:
        lang = 'English'

    nltk.download('punkt')

    if lang == 'Spanish':
        nltk.download('perluniprops')
        nltk.download('nonbreaking_prefixes')
        from nltk.tokenize.toktok import ToktokTokenizer
        toktok = ToktokTokenizer()

    owd = os.getcwd()
    os.chdir(os.path.join(config.get_syntaxnet_folder(), "research", "syntaxnet"))

    if lang == 'Spanish':
        string = sentence.decode('utf-8')
        tokens = toktok.tokenize(string)
        string = " ".join(tokens).encode('utf-8')
        command = 'echo "%s" | syntaxnet/models/parsey_universal/parse.sh syntaxnet/models/Spanish'%(string)
        output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
    if lang == 'English':
        string = sentence
        try:
            tokens = nltk.word_tokenize(string)
            string = " ".join(tokens)
        except:
            tokens = nltk.word_tokenize(string.decode('utf-8'))
            string = " ".join(tokens).encode('utf-8')
        command = 'echo "%s" | syntaxnet/demo.sh'%(string)
        output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
    os.chdir(owd)

    return output.decode()


def pos_tree_tagging(databaseFolder, lang=None):

    if lang == None:
        lang = 'English'

    nltk.download('punkt')

    if lang == 'Spanish':
        nltk.download('perluniprops')
        nltk.download('nonbreaking_prefixes')
        from nltk.tokenize.toktok import ToktokTokenizer
        toktok = ToktokTokenizer()

    owd = os.getcwd()
    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    os.chdir(os.path.join(owd,"text","models","research","syntaxnet"))

    for className in classes:
        fileNames = sorted([f for f in os.listdir(os.path.join(databaseFolder, className))
                            if os.path.isfile(os.path.join(databaseFolder, className, f))
                            and not f.startswith('.') and f[-4:].lower() == ".txt"],
                           key=lambda f: f.lower())
        if not os.path.exists(os.path.join(databaseFolder, className, "POS_trees")):
            os.makedirs(os.path.join(databaseFolder, className, "POS_trees"))
        for name in fileNames:
            if lang == 'Spanish':
                string = codecs.open(os.path.join(databaseFolder, className, name), 'r', 'utf-8').read()
                tokens = toktok.tokenize(string)
                string = " ".join(tokens).encode('utf-8')
                command = 'echo "%s" | syntaxnet/models/parsey_universal/parse.sh syntaxnet/models/Spanish'%(string)
                output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
            elif lang == 'English':
                with open(os.path.join(databaseFolder, className, name)) as file:
                    string = file.readline()
                try:
                    tokens = nltk.word_tokenize(string)
                    string = " ".join(tokens)
                except:
                    tokens = nltk.word_tokenize(string.decode('utf-8'))
                    string = " ".join(tokens).encode('utf-8')
                command = 'echo "%s" | syntaxnet/demo.sh'%(string)
                output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()

            with open(os.path.join(databaseFolder, className, "POS_trees", name), "w+") as file:
                file.write(output)
    os.chdir(owd)


def pre_clean(datasetsFolder):

    classes = sorted([f for f in os.listdir(datasetsFolder)
                      if os.path.isdir(os.path.join(datasetsFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    for className in classes:
        preprocess.cleanText(os.path.join(datasetsFolder, className))
    print("Texts cleaned.")


def tag_count(df):
    tag_count = []
    tags = load_tags()
    for tag in sorted(tags):
        df_tags = df['UPOS'].values.tolist()
        count = df_tags.count(tag)
        tag_count.append(count)
    return tag_count