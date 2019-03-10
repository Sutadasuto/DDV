import os

import keras_tools.lstm as lstm
import tools.subject_analysis as sa

from tools import config

# folder = "/media/winbuntu/google-drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/covarep_features"
# folder = "/media/sutadasuto/OS/Users/Sutadasuto/Google Drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/covarep_features"
# folder = "/media/winbuntu/google-drive/INAOE/Thesis/SpanishDatabase/Aborto_Amigo_/covarep_features"

dataset_name = "court_pruned"
print("\n****\nWorking the %s database\n****\n" % dataset_name)
database_folder, plot_title, transcripts_folder, audios_folder, \
of_target_folder, covarep_target_folder, datasets_folder, complementarity_folder \
    = config.config_database_variables(dataset_name)

folds = 10
custom_folds, custom_dicts = sa.get_cross_iterable(
    sa.get_dict(os.path.join(database_folder, "subjects.txt")),
    folds, processedDataFolder=datasets_folder
)

# lstm.test()
# my_lstm = lstm.basic_binary_lstm_cv(folder)
lstm.standard_vs_binary(of_target_folder, custom_folds)