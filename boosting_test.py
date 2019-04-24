import os
import csv
import numpy as np
from shutil import copyfile
from joblib import Parallel, delayed
import time
import multiprocessing

from audio import audio_analysis as a
from video import video_analysis as v
from text import text_analysis as t
from tools import config
import tools.prune as prune
import tools.subject_analysis as sa
import tools.arff_and_matrices as am
import tools.machine_learning as ml
import tools.multimodal_fusion as fusion
import tools.multimodal_fusion_2 as fusion2

from sklearn.model_selection import LeaveOneOut

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

loo = LeaveOneOut()

clf = LinearSVC(random_state=10, tol=1e-7, max_iter=3000)

datasets_folder = "/media/sutadasuto/OS/Users/Sutadasuto/Google Drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/datasets/visual"
data = "au_intensity"
subjects_dict = "/media/sutadasuto/OS/Users/Sutadasuto/Google Drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips/subjects.txt"
num_folds = 10

custom_folds, custom_dicts = sa.get_cross_iterable(
    sa.get_dict(subjects_dict),
    num_folds, processedDataFolder=datasets_folder
)

files_list = sorted([os.path.join(data, f) for f in os.listdir(os.path.join(datasets_folder, data))
                          if os.path.isfile(os.path.join(datasets_folder, data, f)) and not f.startswith('.')
                            and f.endswith(".arff")], key=lambda f: f.lower())
matrix_1 = ml.complementarity_analysis(clf, datasets_folder, files_list, folds=custom_folds)
matrix_2 = ml.complementarity_analysis(clf, datasets_folder, [data + ".arff"], folds=custom_folds)
matrix_3 = ml.my_method(
                fusion.S4DB(clf, clf, data, datasets_folder),
                datasets_folder, custom_folds, "bssd_views"
            )
a=0