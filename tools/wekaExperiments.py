# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# wekaExperiments.py
# Copyright (C) 2014-2015 Fracpete (pythonwekawrapper at gmail dot com)

from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.classifiers import Evaluation
from weka.core.classes import Random

class Experiment:

    def __init__(self):
        self.header = []
        self.values = []

    def runCV(this, arffFile, classifier, folds):

        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(arffFile)
        data.class_is_last()

        classes = [str(code) for code in data.class_attribute.values]
        header = ["Accuracy"]
        for name in classes:
            header += [name + " TP", name + " FP", name + " AUC ROC"]
        values = []

        cls = Classifier(classname=classifier)

        evl = Evaluation(data)
        evl.crossvalidate_model(cls, data, folds, Random(1))

        values.append(evl.percent_correct)
        for name in classes:
            index = classes.index(name)
            values += [evl.true_positive_rate(index) * 100, evl.false_positive_rate(index) * 100,
                       evl.area_under_roc(index)]

        this.values = values
        this.header = header

    def crossTest(this, trainingFile, classifier, testFile):

        loader = Loader(classname="weka.core.converters.ArffLoader")
        data1 = loader.load_file(trainingFile)
        data1.class_is_last()

        cls = Classifier(classname=classifier)
        cls.build_classifier(data1)

        data2 = loader.load_file(testFile)
        data2.class_is_last()

        classes = [str(code) for code in data2.class_attribute.values]
        header = ["Accuracy"]
        for name in classes:
            header += [name + " TP", name + " FP", name + " AUC ROC"]
        values = []

        evl = Evaluation(data2)
        evl.test_model(cls, data2)

        values.append(evl.percent_correct)
        for name in classes:
            index = classes.index(name)
            values += [evl.true_positive_rate(index) * 100, evl.false_positive_rate(index) * 100,
                       evl.area_under_roc(index)]

        this.values = values
        this.header = header

    def train_and_predict_instances(self, trainingFile, classifier):

        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(trainingFile)
        data.class_is_last()
        classes = [str(code) for code in data.class_attribute.values]
        head = [className + " probability" for className in classes]
        head.append("Guess")

        cls = Classifier(classname=classifier)
        cls.build_classifier(data)

        predictions = [[0,0]] * len(data)
        realLabels = [""] * len(data)
        guess = [0] * len(data)

        for index, inst in enumerate(data):
            pred = cls.classify_instance(inst)
            if inst.get_value(inst.class_index) == pred:
                guess[index] = 1.0
            else:
                guess[index] = 0.0
            dist = cls.distribution_for_instance(inst)
            predictions[index] = [p for p in dist]
            realLabels[index] = classes[int(inst.get_value(inst.class_index))]
            print(str(index + 1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))

        return [predictions, guess, head, realLabels]

    def train_and_separate_validation(self, trainingSet, validationSet, validationInstancesNames, classifier):

        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(trainingSet)
        data.class_is_last()
        data2 = loader.load_file(validationSet)
        if not len(data2) == len(validationInstancesNames):
            print("Theres a mismatch between the number of instances in the arff file and the list of instance names.")
            raise LookupError
        data2.class_is_last()
        classes = [str(code) for code in data.class_attribute.values]
        header = [[classifier, trainingSet, "","",""], ["Instance"] + [className + " probability" for className in classes] + ["Real Class", "Guess"]]

        cls = Classifier(classname=classifier)
        print("Training.")
        cls.build_classifier(data)
        print("Model done!")

        dataMatrix = [["",0,0,0,""] for i in range(len(data2))]

        print("Validating.")
        for index, inst in enumerate(data2):
            print("Instance: " + str(index+1) + "/" + str(len(data2)))
            pred = cls.classify_instance(inst)
            if inst.get_value(inst.class_index) == pred:
                guessValue = 1.0
            else:
                guessValue = 0.0
            dist = cls.distribution_for_instance(inst)
            dataMatrix[index][0] = validationInstancesNames[index]
            dataMatrix[index][1:3] = [round(p,2) for p in dist]
            dataMatrix[index][3] = classes[int(inst.get_value(inst.class_index))]
            dataMatrix[index][4] = guessValue

        print("Done\n")
        return [header, dataMatrix]