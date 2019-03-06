import math
import os
import numpy as np

def get_dict(subjectsFilePath=None):
    if subjectsFilePath == None:
        subjectsFilePath = "subjects.txt"
    subjects_dict = {}

    with open(subjectsFilePath) as subjects_file:
        lines = subjects_file.readlines()
        add_flag = False
        current_subject = ""
        for line in lines:
            if not line.startswith("#"):
                if "subject " in line and add_flag == False:
                    current_subject = line.strip()
                    files = []
                    add_flag = True
                elif not "subject" in line:
                    files.append(line.strip().split(".")[0])
                elif "subject" in line and add_flag == True:
                    subjects_dict[current_subject] = files
                    current_subject = line.strip()
                    files = []
        subjects_dict[current_subject] = files
    return subjects_dict


def get_cross_iterable(subjects_dict, folds, seed=None, processedDataFolder=None):

    if processedDataFolder == None:
        processedDataFolder = "datasets"
    if seed == None:
        seed = folds

    try:
        with open(os.path.join(processedDataFolder, "list_of_instances.csv")) as f:
            lines = f.readlines()
    except:
        raise

    col1 = [line.split(",")[0] for line in lines]
    col2 = [line.split(",")[1].strip() for line in lines]
    for key in list(subjects_dict.keys()):
        subject_exists = False
        instances = []
        for element in subjects_dict[key]:
            if element in col1:
                subject_exists = True
                index = col1.index(element)
                label = col2[index]
                instances.append((element, label))
                col1.pop(index)
                col2.pop(index)
        if subject_exists:
            subjects_dict[key] = instances
        else:
            del subjects_dict[key]

    subject_nums = list(subjects_dict.keys())
    if len(subject_nums) < folds:
        print("Number of subjects can't be lower than the number of folds.")
        print("Folds: %s, Subjects: %s" % (folds, len(subject_nums)))
        raise ValueError
    subject_nums.sort()
    stratified = False
    while not stratified:
        np.random.RandomState(seed).shuffle(subject_nums)
        step = int(math.floor(len(subject_nums) / folds))
        packs = []

        for fold in range(folds):
            pack = []
            for position in range(step):
                index = fold * step + position
                try:
                    pack.append(subject_nums[index])
                except:
                    break
            packs.append(pack)

        if (index+1) < len(subject_nums):
            for position in range(index+1, len(subject_nums)):
                packs[-1].append(subject_nums[position])
        while len(packs[-1]) > int(math.floor(len(subject_nums) / folds)):
            for i in range(len(packs[-1]) - int(math.floor(len(subject_nums) / folds))):
                if i < (folds - 1):
                    packs[i].append(packs[-1].pop())

        stratified = True
        for pack in packs:
            labels = []
            for subject in pack:
                for instance in subjects_dict[subject]:
                    labels.append(instance[1])
            num_labels = len(set(labels))
            if num_labels == 1:
                stratified = False
                break


    trainingSets = []
    testSets = []
    training_subjects_dicts = []
    test_subjects_dicts = []
    for fold in range(folds):
        trainingSubjects = []

        for index in range(folds):
            if index == fold:
                testSubjects = packs[index]
            else:
                trainingSubjects += packs[index]

        trainingInstances = []
        sample = 0
        training_subjects_dict = {}
        for subject in trainingSubjects:
            trainingInstances += subjects_dict.get(subject)
            training_subjects_dict[subject] = [(i, subjects_dict.get(subject)[i-sample][1]) for i in range(sample, sample + len(subjects_dict.get(subject)))]
            sample = sample + len(subjects_dict.get(subject))
        training_subjects_dicts.append(training_subjects_dict)

        testInstances = []
        sample = 0
        test_subjects_dict = {}
        for subject in testSubjects:
            testInstances += subjects_dict.get(subject)
            test_subjects_dict[subject] = [(i, subjects_dict.get(subject)[i-sample][1]) for i in range(sample, sample + len(subjects_dict.get(subject)))]
            sample = sample + len(subjects_dict.get(subject))
        test_subjects_dicts.append(test_subjects_dict)

        trainingIndices = []
        testIndices = []
        try:
            with open(os.path.join(processedDataFolder, "list_of_instances.csv")) as f:
                lines = f.readlines()
        except:
            raise

        lines = [line.split(",")[0] for line in lines]
        for instance in trainingInstances:
            try:
                trainingIndices.append(lines.index(instance[0]))
            except:
                print(instance[0] + " not in list of instances.")
        for instance in testInstances:
            try:
                testIndices.append(lines.index(instance[0]))
            except:
                print(instance[0] + " not in list of instances.")

        trainingSets.append(trainingIndices)
        testSets.append(testIndices)

    customFolds = [() for i in range(folds)]
    customDicts = [() for i in range(folds)]
    for i in range(folds):
        customFolds[i] = (np.array(trainingSets[i]), np.array(testSets[i]))
        customDicts[i] = (training_subjects_dicts[i], test_subjects_dicts[i])
    return customFolds, customDicts


def get_nested_cross_iterable(custom_dict, folds, seed=None, return_custom_dict=False):

    if seed is None:
        seed = folds

    subject_nums = list(custom_dict.keys())
    if len(subject_nums) < folds:
        print("Number of subjects can't be lower than the number of folds.")
        print("Folds: %s, Subjects: %s" % (folds, len(subject_nums)))
        raise ValueError
    subject_nums.sort()
    stratified = False
    while not stratified:
        np.random.RandomState(seed).shuffle(subject_nums)
        step = int(math.floor(len(subject_nums) / folds))
        packs = []

        for fold in range(folds):
            pack = []
            for position in range(step):
                index = fold * step + position
                try:
                    pack.append(subject_nums[index])
                except:
                    break
            packs.append(pack)

        if (index + 1) < len(subject_nums):
            for position in range(index + 1, len(subject_nums)):
                packs[-1].append(subject_nums[position])
        while len(packs[-1]) > int(math.floor(len(subject_nums) / folds)):
            for i in range(len(packs[-1]) - int(math.floor(len(subject_nums) / folds))):
                if i < (folds - 1):
                    packs[i].append(packs[-1].pop())

        stratified = True
        for pack in packs:
            labels = []
            for subject in pack:
                for instance in custom_dict[subject]:
                    labels.append(instance[1])
            num_labels = len(set(labels))
            if num_labels == 1:
                stratified = False
                break

    training_sets = []
    test_sets = []
    training_subjects_dicts = []
    test_subjects_dicts = []
    for fold in range(folds):
        training_subjects = []

        for index in range(folds):
            if index == fold:
                test_subjects = packs[index]
            else:
                training_subjects += packs[index]

        training_instances = []
        sample = 0
        training_subjects_dict = {}
        for subject in training_subjects:
            training_instances += [instance[0] for instance in custom_dict.get(subject)]
            if return_custom_dict:
                training_subjects_dict[subject] = [(i, custom_dict.get(subject)[i-sample][1]) for i in range(sample, sample + len(custom_dict.get(subject)))]
                sample = sample + len(custom_dict.get(subject))
        if return_custom_dict:
            training_subjects_dicts.append(training_subjects_dict)

        test_instances = []
        sample = 0
        test_subjects_dict = {}
        for subject in test_subjects:
            test_instances += [instance[0] for instance in custom_dict.get(subject)]
            if return_custom_dict:
                test_subjects_dict[subject] = [(i, custom_dict.get(subject)[i-sample][1]) for i in range(sample, sample + len(custom_dict.get(subject)))]
                sample = sample + len(custom_dict.get(subject))
        if return_custom_dict:
            test_subjects_dicts.append(test_subjects_dict)

        training_sets.append(training_instances)
        test_sets.append(test_instances)

    custom_folds = [() for i in range(folds)]
    if return_custom_dict:
        custom_dicts = [() for i in range(folds)]
    for i in range(folds):
        custom_folds[i] = (np.array(training_sets[i]), np.array(test_sets[i]))
        if return_custom_dict:
            custom_dicts[i] = (training_subjects_dicts[i], test_subjects_dicts[i])

    if return_custom_dict:
        return custom_folds, custom_dicts
    else:
        return custom_folds