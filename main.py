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

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

print("Libraries loaded.")
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

beginning = time.time()
classifierList = [
    LinearSVC(random_state=10, tol=1e-7, max_iter=3000),
    # DecisionTreeClassifier(random_state=10),
    # RandomForestClassifier(n_estimators=50, random_state=10),
    # AdaBoostClassifier(DecisionTreeClassifier(random_state=10), random_state=10),
    GaussianNB(),
    # LogisticRegression()
]
print("List of classifiers ready.")

dataset_name = "court_pruned"
fps = 29.97
print("Working the %s database" % dataset_name)
database_folder, plot_title, transcripts_folder, audios_folder, \
of_target_folder, covarep_target_folder, datasets_folder, complementarity_folder \
    = config.config_database_variables(dataset_name)
print("Dataset parameters loaded.")

extract_audio = True
extract_transcripts = True

covarep = True
openFace = True
time_stamped_pos = True

audio_analysis = True
text_analyzer = True
video_analysis = True
ngrams = [1, 2, 3, 4]
# ngrams = [1, 2, 3]

mixup = True
early_fusions = True

analysis = False
features_analysis = False
fusion_analysis = False
complementariy_analysis = False

boosting_comparison = False

folds = 10

try:
    # Here we convert the .mp4 files to .wav audios
    if extract_audio:
        a.videos_to_audio(database_folder, audios_folder)
except:
    print("ERROR extracting audios.")
    raise

try:
    if covarep:
        a.extract_features_covarep(audios_folder, covarep_target_folder, 1/fps)
except:
    print("ERROR with COVAREP.")
    raise

try:
    if extract_transcripts:
        a.transcript_files(audios_folder, transcripts_folder,
                           language='English', multispeaker=False)
except:
    print("Error while transcribing.")
    raise

try:
    if time_stamped_pos:
        t.get_timestamped_pos_tags(transcripts_folder)
except:
    print("Error while extracting linguistical features.")

try:
    if openFace:
        v.extract_features(database_folder, of_target_folder)
except:
    print("ERROR extracting video features.")
    raise

prune.clear_error_log()
try:
    if video_analysis:
        print("Getting views from visual modality.")
        v.get_statistics(of_target_folder, os.path.join(datasets_folder, "visual"))
        v.get_statistics_per_category(of_target_folder, os.path.join(datasets_folder, "visual"))
except:
    print("ERROR in video modality!")
    raise

try:
    if audio_analysis:
        print("Getting views from accoustical modality.")
        a.get_statistics_covarep(covarep_target_folder, os.path.join(datasets_folder, "accoustic"))
        a.get_statistics_per_category(covarep_target_folder, os.path.join(datasets_folder, "accoustic"))
except:
    print("ERROR extracting accoustic features.")
    raise

try:
    if text_analyzer:
        print("Getting views from textual modality.")
        t.get_features_per_category(transcripts_folder, targetFileFolder=os.path.join(datasets_folder, "textual"), n=ngrams)
        fusion.early_fusion(os.path.join(datasets_folder, "textual"),
                            fileNames=sorted([f for f in os.listdir(os.path.join(datasets_folder, "textual"))
                                              if os.path.isfile(os.path.join(datasets_folder, "textual", f))
                                              and not f.startswith('.') and f.endswith(".arff") and f != "all.arff"],
                                             key=lambda f: f.lower()),
                            targetFileFolder=os.path.join(datasets_folder, "textual"), outputFileName="all",
                            relation="all_linguistical")
except:
    print("ERROR in text modality!")
    raise

if prune.confirm_files_per_modality(datasets_folder):
    # prune.prune_conflictive_files(datasets_folder)
    print("All views processed from the same files.")
else:
    print("There was a mismatch of the files analyzed per modality.")
    raise RuntimeError

if mixup:

    if not os.path.exists(os.path.join(datasets_folder, "all_views")):
        os.makedirs(os.path.join(datasets_folder, "all_views"))
    if not os.path.exists(os.path.join(datasets_folder, "all_modalities")):
        os.makedirs(os.path.join(datasets_folder, "all_modalities"))
    modalities_to_mix = ["visual", "accoustic", "textual"]

    for modality in modalities_to_mix:
        copyfile(os.path.join(datasets_folder, modality, "all.arff"),
                 os.path.join(datasets_folder, "all_modalities", "%s.arff" % (modality))
                 )
        views = sorted([f for f in os.listdir(os.path.join(datasets_folder, modality))
                        if os.path.isfile(os.path.join(datasets_folder, modality, f)) and not f.startswith('.')
                        and f.endswith(".arff") and f != "all.arff"], key=lambda f: f.lower())
        for view in views:
            copyfile(os.path.join(datasets_folder, modality, view),
                     os.path.join(datasets_folder, "all_views", view)
                     )

if early_fusions:
    fusion.early_fusion(datasets_folder,
                        fileNames=["visual/all.arff",
                                   "accoustic/all.arff",
                                   "textual/all.arff"],
                        targetFileFolder=os.path.join(datasets_folder, "fusion"), relation="early_fusion")
    if mixup:
        copyfile(os.path.join(datasets_folder, "fusion", "early_fusion.arff"),
                 os.path.join(datasets_folder, "all_views", "all.arff"))
        copyfile(os.path.join(datasets_folder, "fusion", "early_fusion.arff"),
                 os.path.join(datasets_folder, "all_modalities", "all.arff"))
    # fusion.au_informed_liwc(of_target_folder, transcripts_folder, os.path.join(datasets_folder, "fusion"))
    fusion.syntax_informed(transcripts_folder, covarep_target_folder, os.path.join(datasets_folder, "fusion"))
    fusion.au_informed(of_target_folder, covarep_target_folder, os.path.join(datasets_folder, "fusion"))
    # fusion.syntax_informed_au(transcripts_folder, of_target_folder, os.path.join(datasets_folder, "fusion"))

custom_folds, custom_dicts = sa.get_cross_iterable(
    sa.get_dict(os.path.join(database_folder, "subjects.txt")),
    folds, processedDataFolder=datasets_folder
)
new_folds, new_dicts = sa.get_nested_cross_iterable(custom_dicts[0][0], folds, 10, True)
# custom_folds = 10
visualFiles = sorted([os.path.join("visual", f) for f in os.listdir(os.path.join(datasets_folder, "visual"))
                      if os.path.isfile(os.path.join(datasets_folder, "visual", f))
                      and not f.startswith('.') and f.endswith(".arff") and f != "all.arff"],
                     key=lambda f: f.lower())
accousticFiles = sorted([os.path.join("accoustic", f) for f in os.listdir(os.path.join(datasets_folder, "accoustic"))
                         if os.path.isfile(os.path.join(datasets_folder, "accoustic", f))
                         and not f.startswith('.') and f.endswith(".arff") and f != "all.arff"],
                        key=lambda f: f.lower())
linguisticalFiles = sorted([os.path.join("textual", f) for f in os.listdir(os.path.join(datasets_folder, "textual"))
                            if os.path.isfile(os.path.join(datasets_folder, "textual", f))
                            and not f.startswith('.') and f.endswith(".arff") and f != "all.arff"],
                           key=lambda f: f.lower())
timeFusionFiles = sorted([os.path.join("fusion", f) for f in os.listdir(os.path.join(datasets_folder, "fusion"))
                          if os.path.isfile(os.path.join(datasets_folder, "fusion", f))
                          and not f.startswith('.') and f.endswith(".arff") and "_informed" in f],
                         key=lambda f: f.lower())
fusionFiles = ["fusion/early_fusion.arff"] + timeFusionFiles

modalityFiles = ["visual/all.arff",
                 "accoustic/all.arff",
                 "textual/all.arff"]
files = (visualFiles + accousticFiles + linguisticalFiles)
filesLists = [visualFiles, ["visual/all.arff"], accousticFiles, ["accoustic/all.arff"], linguisticalFiles,
              ["textual/all.arff"], fusionFiles]
# filesLists = [visualFiles, ["all_visual.arff"], accousticFiles, fusionFiles]
cfdDictionary = {tuple(visualFiles): "visual_features",
                 tuple(accousticFiles): "accoustical_features",
                 tuple(linguisticalFiles): "linguistical_features",
                 tuple(modalityFiles): "all_modalities",
                 tuple(files): "all_views"}
header = [""]
mpaList = ["MPA"]
cfdList = ["CFD"]
classifierResults = []
complementarityResults = []
if not os.path.exists(complementarity_folder):
    os.makedirs(complementarity_folder)
if analysis:
    for clf in classifierList:
        matrices = []
        header = [""]
        mpaList = ["MPA"]
        cfdList = ["CFD"]
        if features_analysis:
            for filesList in filesLists:
                matrix = ml.complementarity_analysis(clf, datasets_folder, filesList, folds=custom_folds)
                if cfdDictionary.get(tuple(filesList)) != None:
                    header.append(cfdDictionary.get(tuple(filesList)))
                    if complementariy_analysis:
                        mpa, cfd = ml.coincident_failure_diversity(matrix)
                        mpaList.append(round(mpa, 1))
                        cfdList.append(round(cfd, 3))
                matrices.append(matrix)

        if fusion_analysis:
            matrix = ml.hard_majority_vote_evaluation(clf, datasets_folder, files,
                                                      custom_folds, "majority_vote_views")
            matrices.append(matrix)
            matrix = ml.hard_majority_vote_evaluation(clf, datasets_folder, modalityFiles,
                                                      custom_folds, "majority_vote_modalities")
            matrices.append(matrix)
            matrix = ml.stacking_evaluation(clf, datasets_folder, files,
                                            custom_folds, "stacking_views")
            matrices.append(matrix)
            matrix = ml.stacking_evaluation(clf, datasets_folder, modalityFiles,
                                            custom_folds, "stacking_modalities")
            matrices.append(matrix)
            matrix = ml.stacking_proba_evaluation(clf, datasets_folder, files,
                                                  custom_folds, "stacking_views_with_proba")
            matrices.append(matrix)
            matrix = ml.stacking_proba_evaluation(clf, datasets_folder, modalityFiles,
                                                  custom_folds, "stacking_modalities_with_proba")
            matrices.append(matrix)
            matrix = ml.my_method(
                fusion.S3DB(clf, clf, ["visual", "textual", "accoustic"], datasets_folder,
                            file_exceptions=["all.arff"]),
                datasets_folder, custom_folds, "poster_proposal"
            )
            matrices.append(matrix)
            matrix = ml.my_method(
                fusion.BSSD(clf, "all_views", datasets_folder),
                datasets_folder, custom_folds, "bssd_views"
            )
            matrices.append(matrix)
            matrix = ml.my_method(
                fusion.BSSD(clf, "all_modalities", datasets_folder),
                datasets_folder, custom_folds, "bssd_modalities"
            )
            matrices.append(matrix)
            matrix = ml.my_method(
                fusion.BSSD2(clf, "all_views", datasets_folder),
                datasets_folder, custom_folds, "bssd_views_cv"
            )
            matrices.append(matrix)
            matrix = ml.my_method(
                fusion.BSSD2(clf, "all_modalities", datasets_folder),
                datasets_folder, custom_folds, "bssd_modalities_cv"
            )
            matrices.append(matrix)
            matrix = ml.my_method(
                fusion.BSSD2_2(clf, "all_views", datasets_folder),
                datasets_folder, (custom_folds, custom_dicts), "bssd_views_cv_subject"
            )
            matrices.append(matrix)
            matrix = ml.my_method(
                fusion.BSSD2_2(clf, "all_modalities", datasets_folder),
                datasets_folder, (custom_folds, custom_dicts), "bssd_modalities_cv_subject"
            )
            matrices.append(matrix)
            matrix = ml.my_method(
                fusion.S4DB(clf, clf, "all_views", datasets_folder),
                datasets_folder, custom_folds, "stacking_bssd_views"
            )
            matrices.append(matrix)
            matrix = ml.my_method(
                fusion.S4DB(clf, clf, "all_modalities", datasets_folder),
                datasets_folder, custom_folds, "stacking_bssd_modalities"
            )
            matrices.append(matrix)

        matrix = ml.concatenate_result_matrices(matrices)
        classifierResults.append(
            np.column_stack((np.array([[str(clf).split("(")[0]], [""]]), matrix[-2:, :-1])).transpose())
        firstColumn = np.concatenate((np.array(["", ""]), matrix[0, 1:-1]))
        with open(os.path.join(complementarity_folder, str(clf).split("(")[0] + ".csv"), "w+") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(matrix.tolist())

        am.matrices_comparison(classifierResults, firstColumn, complementarity_folder, plot_title=plot_title)

        if complementariy_analysis:
            matrix = ml.complementarity_analysis(clf, datasets_folder, modalityFiles, folds=custom_folds)
            header.append(cfdDictionary.get(tuple(modalityFiles)))
            mpa, cfd = ml.coincident_failure_diversity(matrix)
            mpaList.append(round(mpa, 1))
            cfdList.append(round(cfd, 3))
            matrix = ml.complementarity_analysis(clf, datasets_folder, files, folds=custom_folds)
            header.append(cfdDictionary.get(tuple(files)))
            mpa, cfd = ml.coincident_failure_diversity(matrix)
            mpaList.append(round(mpa, 1))
            cfdList.append(round(cfd, 3))
            with open(os.path.join(complementarity_folder, str(clf).split("(")[0] + "_complementarity.csv"),
                      "w+") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows([header, cfdList, mpaList])
                complementarityResults.append([str(clf).split("(")[0]] + cfdList)
                complementarityResults.append([""] + mpaList)

            am.complementarity_comparison(complementarityResults, header, complementarity_folder, plot_title=plot_title)

    am.matrices_comparison(classifierResults, firstColumn, complementarity_folder, plot_title=plot_title)
    if complementariy_analysis:
        am.complementarity_comparison(complementarityResults, header, complementarity_folder, plot_title=plot_title)

a = 0


def get_boosting_objects(weak_learner, booster, custom_folds, custom_dicts, datasets_folder, modalities):
    object_lists = []
    object_list = []
    for modality in modalities:
        object_list.append(fusion.S4DB(weak_learner, weak_learner, modality, datasets_folder, file_exceptions=["all.arff"]))
    object_list += [custom_folds, "stacking_bssd"]
    object_lists.append(object_list)
    object_list = []
    for modality in modalities:
        object_list.append(fusion.BSSD2_2(weak_learner, modality, datasets_folder, file_exceptions=["all.arff"]))
    object_list += [(custom_folds, custom_dicts), "bssd_cv_subject"]
    object_lists.append(object_list)
    object_list = []
    for modality in modalities:
        object_list.append(fusion.BSSD2(weak_learner, modality, datasets_folder, file_exceptions=["all.arff"]))
    object_list += [custom_folds, "bssd_cv"]
    object_lists.append(object_list)
    object_list = []
    for modality in modalities:
        object_list.append(fusion.BSSD(weak_learner, modality, datasets_folder, file_exceptions=["all.arff"]))
    object_list += [custom_folds, "bssd"]
    object_lists.append(object_list)
    object_list = ["adaboost"]
    for modality in modalities:
        object_list.append(modality)
    object_list.append(booster)
    object_list.append(custom_folds)
    object_lists.append(object_list)

    return object_lists


def perform_boosting_comparison(object_lists, datasets_folder, modalities):
    matrices = []
    if object_lists[0] != "adaboost":
        for idx, modality in enumerate(modalities):
            matrix = ml.my_method(
                object_lists[idx], datasets_folder, object_lists[-2], object_lists[-1]
            )
            matrices.append(matrix)
        matrix = ml.concatenate_result_matrices(matrices)
        return matrix
    else:
        for idx, modality in enumerate(modalities, 1):
            matrix = ml.complementarity_analysis(
                object_lists[-2], datasets_folder, ["%s/all.arff" % (object_lists[idx])], folds=object_lists[-1]
            )
            matrices.append(matrix)
        matrix = ml.concatenate_result_matrices(matrices)
        return matrix


if boosting_comparison:
    booster = AdaBoostClassifier(DecisionTreeClassifier(random_state=10), random_state=10)
    weak_learner = DecisionTreeClassifier(random_state=10)
    modalities = ["visual", "accoustic", "textual", "all_views"]
    classifierResults = []
    now = time.time()
    matrices = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(perform_boosting_comparison)(object_lists, datasets_folder, modalities) for object_lists in
        get_boosting_objects(weak_learner, booster, custom_folds, custom_dicts, datasets_folder, modalities))
    for idx, matrix in enumerate(matrices):
        if idx != len(matrices) - 1:
            classifierResults.append(
                np.column_stack((np.array([[matrices[idx][0][1]], [""]]), matrix[-2:, :-1])).transpose())
        else:
            classifierResults.append(
                np.column_stack((np.array([[matrices[idx][0][0]], [""]]), matrix[-2:, :-1])).transpose())
    print("Object Parallel process done in %s sec" % (time.time() - now))
    firstColumn = np.concatenate((np.array(["", ""]), np.array(modalities)))
    am.matrices_comparison(classifierResults, firstColumn, complementarity_folder, "bssd_adaboost",
                           plot_title=plot_title, plot_subtitles=["Boosting comparison"], category_end=None
                           )
    # classifierResults = []
    # now = time.time()
    # for modality in modalities:
    #     matrices = []
    #     matrix = ml.my_method(
    #         fusion.S4DB(weak_learner, weak_learner, modality, datasets_folder, file_exceptions=["all.arff"]),
    #         datasets_folder, custom_folds, "stacking_bssd"
    #     )
    #     matrices.append(matrix)
    #     matrix = ml.my_method(
    #         fusion.BSSD2_2(weak_learner, modality, datasets_folder, file_exceptions=["all.arff"]),
    #         datasets_folder, (custom_folds, custom_dicts), "bssd_cv_subject"
    #     )
    #     matrices.append(matrix)
    #     matrix = ml.my_method(
    #         fusion.BSSD2(weak_learner, modality, datasets_folder, file_exceptions=["all.arff"]),
    #         datasets_folder, custom_folds, "bssd_cv"
    #     )
    #     matrices.append(matrix)
    #     matrix = ml.my_method(
    #         fusion.BSSD(weak_learner, modality, datasets_folder, file_exceptions=["all.arff"]),
    #         datasets_folder, custom_folds, "bssd"
    #     )
    #     matrices.append(matrix)
    #     matrix = ml.complementarity_analysis(
    #         booster, datasets_folder, ["%s/all.arff" % (modality)], folds=custom_folds
    #     )
    #     matrices.append(matrix)
    #     matrix = ml.concatenate_result_matrices(matrices)
    #     classifierResults.append(np.column_stack((np.array([[modality], [""]]), matrix[-2:, :-1])).transpose())
    # print("Sequential process done in %s sec" % (time.time() - now))
    #
    # firstColumn = np.concatenate((np.array(["", ""]), np.array(matrix[0, 1:-1])))
    # matrices = []
    # am.matrices_comparison(classifierResults, firstColumn, complementarity_folder, "bssd_adaboost",
    #                        plot_title=plot_title, plot_subtitles=["Boosting comparison"],
    #                        )

a = False
if a:
    matrix = ml.my_method(
        fusion.S4DB(DecisionTreeClassifier(random_state=10), DecisionTreeClassifier(random_state=10), "all_views",
                    datasets_folder, file_exceptions=["all.arff"]),
        datasets_folder, custom_folds, "stacking_bssd"
    )
    matrix2 = ml.my_method(
        fusion2.S4DB(DecisionTreeClassifier(random_state=10), DecisionTreeClassifier(random_state=10), "all_views",
                     datasets_folder, file_exceptions=["all.arff"]),
        datasets_folder, custom_folds, "stacking_bssd"
    )
    matrix = ml.my_method(
        fusion.BSSD2_2(DecisionTreeClassifier(random_state=10), "all_views", datasets_folder, file_exceptions=["all.arff"]),
        datasets_folder, (custom_folds, custom_dicts), "bssd_cv_subject"
    )
    matrix2 = ml.my_method(
        fusion2.BSSD2_2(DecisionTreeClassifier(random_state=10), "all_views", datasets_folder, file_exceptions=["all.arff"]),
        datasets_folder, (custom_folds, custom_dicts), "bssd_cv_subject"
    )
    matrix = ml.my_method(
        fusion.BSSD2(DecisionTreeClassifier(random_state=10), "all_views", datasets_folder, file_exceptions=["all.arff"]),
        datasets_folder, custom_folds, "bssd_cv"
    )
    matrix2 = ml.my_method(
        fusion2.BSSD2(DecisionTreeClassifier(random_state=10), "all_views", datasets_folder, file_exceptions=["all.arff"]),
        datasets_folder, custom_folds, "bssd_cv"
    )
    matrix = ml.my_method(
        fusion.S3DB(GaussianNB(), GaussianNB(), ["visual", "textual", "accoustic"], datasets_folder,
                    file_exceptions=["all.arff"]),
        datasets_folder, custom_folds, "poster_proposal"
    )
    matrix2 = ml.my_method(
        fusion2.S3DB(GaussianNB(), GaussianNB(), ["visual", "textual", "accoustic"], datasets_folder,
                     file_exceptions=["all.arff"]),
        datasets_folder, custom_folds, "poster_proposal"
    )
    matrix = ml.my_method(
        fusion.BSSD(DecisionTreeClassifier(random_state=10), "all_views", datasets_folder, file_exceptions=["all.arff"]),
        datasets_folder, custom_folds, "bssd"
    )
    matrix2 = ml.my_method(
        fusion2.BSSD(DecisionTreeClassifier(random_state=10), "all_views", datasets_folder, file_exceptions=["all.arff"]),
        datasets_folder, custom_folds, "bssd"
    )
a = 0
print("Finished %s dataset in %s sec" % (dataset_name, round(time.time()-beginning, 1)))
