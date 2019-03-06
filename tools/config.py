import yaml #pyyaml
import os

def config_database_variables(dataset_name):
    dataset_dict = yaml.load(open("config/%s/dataset.yaml" % (dataset_name)).read())
    database_folder = dataset_dict["database_folder"]
    plot_title = dataset_dict["plot_title"]

    targets_dict = yaml.load(open("config/extraction_targets.yaml").read())
    transcripts_folder = "%s_/%s" % (database_folder, targets_dict["transcripts_folder"])
    audios_folder = "%s_/%s" % (database_folder, targets_dict["audios_folder"])
    of_target_folder = "%s_/%s" % (database_folder, targets_dict["of_target_folder"])
    covarep_target_folder = "%s_/%s" % (database_folder, targets_dict["covarep_target_folder"])
    datasets_folder = "%s_/%s" % (database_folder, targets_dict["datasets_folder"])
    complementarity_folder = "%s_/%s" % (database_folder, targets_dict["complementarity_folder"])

    return  database_folder, plot_title, transcripts_folder, audios_folder, of_target_folder, \
            covarep_target_folder, datasets_folder, complementarity_folder

def get_openface_folder():
    dict = yaml.load(open("config/openface.yaml"))
    return dict["openface_folder"]

def get_syntaxnet_folder():
    dict = yaml.load(open("config/syntaxnet.yaml"))
    return dict["syntaxnet_folder"]