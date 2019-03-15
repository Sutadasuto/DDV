import os

import keras_tools.lstm as lstm
import keras_tools.blstm as blstm
import tools.subject_analysis as sa
import video.video_analysis as video
import audio.audio_analysis as audio

from tools import config

# folder = "/media/winbuntu/google-drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/covarep_features"
# folder = "/media/sutadasuto/OS/Users/Sutadasuto/Google Drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/covarep_features"
# folder = "/media/winbuntu/google-drive/INAOE/Thesis/SpanishDatabase/Aborto_Amigo_/covarep_features"

dataset_name = "court_full"
print("\n****\nWorking the %s database\n****\n" % dataset_name)
database_folder, plot_title, transcripts_folder, audios_folder, \
of_target_folder, covarep_target_folder, datasets_folder, complementarity_folder \
    = config.config_database_variables(dataset_name)

folds = 10
custom_folds, custom_dicts = sa.get_cross_iterable(
    sa.get_dict(os.path.join(database_folder, "subjects.txt")),
    folds, processedDataFolder=datasets_folder
)

if not os.path.exists(os.path.join(os.path.split(of_target_folder)[0], "of_frames")):
    video.get_frames_per_category(of_target_folder)

if not os.path.exists(os.path.join(os.path.split(covarep_target_folder)[0], "covarep_frames")):
    audio.get_frames_per_category(covarep_target_folder)

# lstm.test()
# my_lstm = lstm.basic_binary_lstm_cv(folder)
# lstm.modalities((covarep_target_folder, of_target_folder), custom_folds)
blstm.modalities((covarep_target_folder, of_target_folder), custom_folds)
# lstm.standard_vs_binary(of_target_folder, custom_folds)
