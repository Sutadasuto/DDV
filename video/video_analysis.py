import os
import subprocess
import numpy as np
import scipy.stats
import pandas
import tools.arff_and_matrices as am
from tools import config


def extract_features(databaseFolder, outputFolder):

    owd = os.getcwd()
    categories = sorted([f for f in os.listdir(databaseFolder)
                           if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                         key=lambda f: f.lower())

    for category in categories:

        videos = sorted([f for f in os.listdir(databaseFolder + "/" + category)
                           if os.path.isfile(os.path.join(databaseFolder + "/" + category, f)) and not
                         f.startswith('.')], key=lambda f: f.lower())
        command = ""
        for video in videos:
            command += ' -f "'
            command += '%s/%s/%s"'%(databaseFolder, category, video)
            #command += databaseFolder + "/" + category + "/" + video + '"'

        os.chdir(config.get_openface_folder())
        command = 'build/bin/FeatureExtraction%s -out_dir "%s"'%(command, os.path.join(outputFolder,category))
        command += " -2Dfp -pose -aus -gaze"
        subprocess.call(command, shell=True)
        os.chdir(owd)
    print("OpenFace analysis complete.")

def get_frames_per_category(databaseFolder, processedDataFolder=None):

    if processedDataFolder == None:
        processedDataFolder = "datasets/visual"

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    categoryDictionary = {"gaze": ["gaze_"],
                          "eye_landmarks": ["eye_lmk_"],
                          "head": ["pose_"],
                          "facial_landmarks": ["x_", "y_"],
                          "au_intensity": ["_r"],
                          "au_presence": ["_c"]
                          }
    timestamp_label= " timestamp"
    for category in categoryDictionary.keys():
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
                header = []
                df = pandas.read_csv(os.path.join(databaseFolder, className, feat_file), header='infer')
                feature_names = df.columns.values
                for feat in feature_names:
                    reference = categoryDictionary.get(category)
                    for string in reference:
                        if feat.strip().lower().startswith(string) \
                                or feat.strip().lower().endswith(string):
                            header.append(feat)
                df1 = df[header]
                df1.insert(0, "time", df[timestamp_label].values)
        print("Analysis of %s acquired." % (category))
        with open(os.path.join(processedDataFolder, "%s.txt"%(category)), "w+") as files:
            files.write("\n".join(analyzedFiles))

def get_statistics(databaseFolder, processedDataFolder=None, outputFileName=None, relationName=None):

    if processedDataFolder == None:
        processedDataFolder = "datasets/visual"
    if outputFileName== None:
        outputFileName = "all"
    if relationName == None:
        relationName = "all_visual"

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
            for feat in feature_names[5:]:
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
    print("Analysis of all OpenFace features acquired.")
    with open(os.path.join(processedDataFolder, outputFileName + ".txt"), "w+") as files:
        files.write("\n".join(analyzedFiles))

def get_statistics_per_category(databaseFolder, processedDataFolder=None):

    if processedDataFolder == None:
        processedDataFolder = "datasets/visual"

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    stats_names = ['max', 'min', 'mean', 'median', 'std', 'var', 'kurt', 'skew', 'percentile25', 'percentile50',
                   'percentile75']

    categoryDictionary = {"gaze": ["gaze_"],
                          "eye_landmarks": ["eye_lmk_"],
                          "head": ["pose_"],
                          "facial_landmarks": ["x_", "y_"],
                          "au_intensity": ["_r"],
                          "au_presence": ["_c"]
                          }
    for category in categoryDictionary.keys():
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
                    reference = categoryDictionary.get(category)
                    for string in reference:
                        if feat.strip().lower().startswith(string) \
                                or feat.strip().lower().endswith(string):
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
                            break
                if startFlag:
                    matrix = [mm_names + ["Class"]]
                    startFlag = False
                matrix.append(mm_feats + [className])
        am.create_arff(matrix,classes,processedDataFolder,category,category)
        print("Analysis of %s acquired." % (category))
        with open(os.path.join(processedDataFolder, "%s.txt"%(category)), "w+") as files:
            files.write("\n".join(analyzedFiles))