def tokenize(textFile):

    punctuation = [".", ",", ":", ";", "!", "?", "..."]

    lines = textFile.readlines()
    tokens = []
    for line in lines:
        line = line.split("\n")[0]
        tokenList = line.split(" ")

        for element in tokenList:
            if element != '':

                flag = True
                for sign in punctuation:
                    if element[-1] == sign and element != "..." and len(element) > 1:
                        element = element[0:-1]

                for sign in punctuation:
                    parts = element.split(sign)
                    if len(parts) > 1:
                        flag = False
                        for part in parts:
                            if part != '':
                                tokens.append(part.lower())

                if flag:
                    tokens.append(element.lower())
    return tokens


def vectorize(tokens, dict):

    vector = [0] * len(dict.code)
    numTokens = len(tokens)

    for token in tokens:

        if token == "%hesitation":
            index = dict.vector.index(1000)
            vector[index] += 1
        else:

            for key,value in dict.liwcDict.items():

                flag = True
                string = ""
                for k in key:

                    if k == "*":
                        break
                    string += k

                    if string not in token[0:len(string)]:
                        flag = False
                        break

                if key[-1] != "*" and len(string) != len(token):
                    flag = False
                elif key == "*" and token != key:
                    flag = False

                if flag:
                    numbers = value
                    for number in numbers:
                        if number == 1000:
                            a = 0
                        try:
                            index = dict.vector.index(number)
                            vector[index] += 1
                        except:
                            print("Number {} not found in any key").format(number)
                    break
    if numTokens > 0:
        vector = [float(i) / numTokens for i in vector]
    return vector


class Dictionary:

    def __init__(self, dictionary):

        self.code = {}
        codes = []
        self.liwcDict = {}
        liwcPairs = []
        self.vector = []
        self.names = []

        dict = open(dictionary, 'r')
        flag = 0

        for line in dict:

            line = line.replace('\r', '')
            if flag == 0:
                if line == "%\n":
                    flag =1

            elif flag == 1:
                if line == "%\n":
                    flag = 2
                else:
                    tuple = line.split("\t")
                    tuple = [tuple[0], tuple[1][0:-1]]
                    self.names.append(tuple[1])
                    codes.append(tuple)

            elif flag == 2:
                pieces = line.split("\t")
                word = pieces[0]
                numbers = pieces[1:]
                try:
                    liwcPairs.append([word, list(map(int, numbers))])
                except:
                    try:
                        pieces = line.split("\t<")
                        word1 = pieces[0]
                        word2 = pieces[0] + " " + pieces[1].split(">")[0]
                        numbers1 = []
                        numbers2 = []
                        for piece in pieces:
                            if piece is not pieces[0]:
                                values = piece.split(">")[1]
                                numbers2.append(list(map(int, values.split("/")))[0])
                                numbers1.append(list(map(int, values.split("/")))[1])
                        liwcPairs.append([word1, numbers1])
                        liwcPairs.append([word2, numbers2])
                    except:
                        if line.split("\t")[0] == "like":
                            word = "like"
                            numbers = [2, 134, 125, 464, 126, 253]
                            liwcPairs.append([word, numbers])


        for code in codes:
            self.code[int(code[0])] = code[1]
            self.vector.append(int(code[0]))

        for liwcPair in liwcPairs:
            self.liwcDict[liwcPair[0]] = liwcPair[1]

        if len(self.liwcDict) == 0:
            raise NameError("Dictionary not constructed. Double check line breaks in your dictionary file.")