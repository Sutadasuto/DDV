def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import os
import tools.subject_analysis as sa
import tools.arff_and_matrices as am
import tools.machine_learning as ml
import tools.multimodal_fusion as fusion

from sklearn.model_selection import LeaveOneOut

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

loo = LeaveOneOut()

clf = LinearSVC(random_state=0, tol=1e-7, max_iter=3000)  # court cv
clf = SVC(random_state=0, tol=1e-7, max_iter=3000, kernel='poly', C=0.01, probability=True)  # spanish
# clf = RandomForestClassifier(random_state=0, n_estimators=100)  # court loo
clf_2 = LinearSVC(random_state=0, tol=1e-7, max_iter=3000)  # both

datasets_folder = "/media/winbuntu/google-drive/INAOE/Thesis/SpanishDatabase/Aborto_Amigo_/datasets"
data = "best_views_cv"
subjects_dict = "/media/winbuntu/google-drive/INAOE/Thesis/SpanishDatabase/Aborto_Amigo/subjects.txt"
num_folds = 10

custom_folds, custom_dicts = sa.get_cross_iterable(
    sa.get_dict(subjects_dict),
    num_folds, processedDataFolder=datasets_folder
)
#custom_folds=42
fusion_name = "all"
files_list = sorted([os.path.join(data, f) for f in os.listdir(os.path.join(datasets_folder, data))
                          if os.path.isfile(os.path.join(datasets_folder, data, f)) and not f.startswith('.')
                            and f.endswith(".arff") and not f.startswith(fusion_name)], key=lambda f: f.lower())
fusion.early_fusion(datasets_folder, files_list, targetFileFolder=os.path.join(datasets_folder, data), outputFileName=fusion_name, relation=fusion_name)
# files_list = [os.path.join(data, f) for f in ["au_presence.arff", "gaze.arff"]]

matrices = []

matrix = ml.complementarity_analysis(clf, datasets_folder, files_list, folds=custom_folds)
matrices.append(matrix)
matrix = ml.complementarity_analysis(clf, datasets_folder, [os.path.join(data, fusion_name + ".arff")], folds=custom_folds)
matrices.append(matrix)
matrix = ml.hard_majority_vote_evaluation(clf, datasets_folder, files_list,
                                                      custom_folds, "majority_vote_views")
matrices.append(matrix)
matrix = ml.proba_majority_vote_evaluation(clf, datasets_folder, files_list,
                                                      custom_folds, "majority_vote_proba_views")
matrices.append(matrix)
matrix = ml.stacking_evaluation(clf, datasets_folder, files_list,
                                            custom_folds, "stacking_views")
matrices.append(matrix)
matrix = ml.stacking_proba_evaluation(clf, datasets_folder, files_list,
                                            custom_folds, "stacking_proba_views")
matrices.append(matrix)
matrix = ml.my_method(
                fusion.BSSD(clf, data, datasets_folder),
                datasets_folder, custom_folds, "bssd_views", False
            )
matrices.append(matrix)
matrix = ml.my_method(
                fusion.S4DB(clf, clf_2, data, datasets_folder),
                datasets_folder, custom_folds, "stacking_bssd_views", False
            )
matrices.append(matrix)
# matrix = ml.my_method(
#                 fusion.BSSD2(clf, data, datasets_folder),
#                 datasets_folder, custom_folds, "bssd_cv_views", False
#             )
# matrices.append(matrix)
matrix = ml.my_method(
                fusion.S3DB(clf, clf_2, ["best_visual_views_cv", "best_acoustic_views_cv"], datasets_folder,
                            file_exceptions=["all.arff"]),
                datasets_folder, custom_folds, "hierarchical_bssd"
            )
matrices.append(matrix)

matrix = ml.concatenate_result_matrices(matrices)
firstColumn = np.concatenate((np.array(["", ""]), matrix[0, 1:-1]))

classifierResults = [
    np.column_stack((np.array([[str(clf).split("(")[0]], [""]]), matrix[-2:, :-1])).transpose()]

am.matrices_comparison(classifierResults, firstColumn, os.getcwd(), plot_title="Spanish Database")
