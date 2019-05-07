####
# Ensuring tensorflow backend
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras_tools.lstm as lstm
import keras_tools.blstm as blstm
import keras_tools.utilities as utilities
import tools.subject_analysis as sa
import video.video_analysis as video
import audio.audio_analysis as audio

from tools import config

# folder = "/media/winbuntu/google-drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/covarep_features"
# folder = "/media/sutadasuto/OS/Users/Sutadasuto/Google Drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/covarep_features"
# folder = "/media/winbuntu/google-drive/INAOE/Thesis/SpanishDatabase/Aborto_Amigo_/covarep_features"

dataset_name = "aborto_amigo"
print("\n****\nWorking the %s database\n****\n" % dataset_name)
database_folder, plot_title, transcripts_folder, audios_folder, \
of_target_folder, covarep_target_folder, datasets_folder, complementarity_folder \
    = config.config_database_variables(dataset_name)

if not os.path.exists(os.path.join(os.path.split(of_target_folder)[0], "of_frames")):
    video.get_frames_per_category(of_target_folder)
if not os.path.exists(os.path.join(os.path.split(covarep_target_folder)[0], "covarep_frames")):
    audio.get_frames_per_category(covarep_target_folder)

visual_views = utilities.get_modality_views(os.path.join(database_folder + "_",
                                                         "of_frames"
                                                         ))
acoustical_views = utilities.get_modality_views(os.path.join(database_folder + "_",
                                                             "covarep_frames"
                                                             ))
hu = 200
dropout = None
epochs = 1
batch_size = 16
gpu = False
seq_reduction_method = "sync_kmeans"
reduction_parameter = 100
feat_standardization = True
folds = 10

custom_folds, custom_dicts = sa.get_cross_iterable(
    sa.get_dict(os.path.join(database_folder, "subjects.txt")),
    folds, processedDataFolder=datasets_folder
)

# blstm.modalities(acoustical_views+visual_views, custom_folds, seq_reduction_method, reduction_parameter, database_folder + "_",
#                 hu, dropout, epochs, batch_size, gpu)
# lstm.modalities(acoustical_views+visual_views, custom_folds, seq_reduction_method, reduction_parameter, database_folder + "_",
#                 hu, dropout, epochs, batch_size, gpu)
# lstm.my_method([acoustical_views, visual_views], custom_folds, seq_reduction_method, reduction_parameter, database_folder + "_",
#                 hu, dropout, epochs, batch_size, gpu, feat_standardization=feat_standardization)

# import video.video_analysis as video
# video.get_statistics_independently("/media/sutadasuto/OS/Users/Sutadasuto/Google Drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/datasets/visual/au_intensity.arff")

# import keras_tools.vgg_face_tools as vgg
#
# vgg.vgg_fine_tuning("/media/winbuntu/google-drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/faces", batch_size=16, verbose=1)

# lstm.views_grid_search([acoustical_views, visual_views], custom_folds, seq_reduction_method, reduction_parameter)
lstm.views_train_features("/media/winbuntu/google-drive/INAOE/Thesis/SpanishDatabase/Aborto_Amigo_/grid_search_results.txt",
                          [acoustical_views, visual_views], seq_reduction_method, reduction_parameter)