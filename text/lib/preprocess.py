# -*- coding: utf-8 -*-
from os import listdir
from os.path import isfile, join
import nltk

def cleanText(dataset):
    fileNames = sorted([f for f in listdir(dataset) if isfile(join(dataset, f))
                        and not f.startswith('.') and f[-4:].lower() == ".txt"],
                       key=lambda f: f.lower())

    punctuation = [".", ",", ":", ";", "!", "?", "..."]
    for fileName in fileNames:

        textFile = open(dataset + "/" + fileName)
        lines = textFile.readlines()
        tokens = []
        for line in lines:
            line = line.split("\n")[0]
            tokenList = line.split(" ")

            for element in tokenList:
                if element != '':

                    flag = True
                    for sign in punctuation:
                        parts = element.split(sign)
                        if len(parts) > 1:
                            flag = False
                            for part in parts:
                                if part != '':
                                    tokens.append(part.lower())
                                    tokens.append(sign)
                                else:
                                    tokens.append(sign)
                            tokens.pop()
                            break

                    if flag:
                        tokens.append(element.lower())

        textFile.close()
        processedFile = open(dataset + "/" + fileName, 'w')
        processedText = ""
        for token in tokens:
            if token == "%hesitation":
                token = "*"
            for sign in punctuation:
                if sign == token:
                    processedText = processedText[0:-1]
                    break
            processedText += (token + " ")
        processedText = processedText[0:-1]
        processedFile.write(processedText)

def tokenize_sentence(sentence, lang=None, punctList=None):

    if lang == None:
        lang = 'English'
    if punctList == None:
        punctList = [';', ':', ',', '.', '...', '``', "''", '¡', '!', '¿', '?']

    if lang == 'Spanish':
        nltk.download('perluniprops')
        nltk.download('nonbreaking_prefixes')
        from nltk.tokenize.toktok import ToktokTokenizer
        toktok = ToktokTokenizer()

    if lang == 'Spanish':
        string = sentence.decode('utf-8')
        tokens = toktok.tokenize(string)
        words = []
        for token in tokens:
            if not token in punctList:
                words.append(token)
    if lang == 'English':
        string = sentence
        try:
            tokens = nltk.word_tokenize(string)

        except:
            tokens = nltk.word_tokenize(string.decode('utf-8'))
        words = []
        for token in tokens:
            if not token in punctList:
                words.append(token)
    return words