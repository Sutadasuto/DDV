from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras.models import model_from_json
from keras.preprocessing import image
from keras_vggface import utils
from keras_vggface.vggface import VGGFace # https://github.com/rcmalli/keras-vggface
from math import ceil as ceil
from sklearn.preprocessing import LabelEncoder

import csv
import numpy as np
import os
import pandas
import scipy.stats
import subprocess
import tools.arff_and_matrices as am
import tools.config as config

np.random.seed(0)


def data_generator(images_folder, batch_size, split_indices, mode):

    img_paths = []
    categories = sorted([f for f in os.listdir(images_folder)
                         if os.path.isdir(os.path.join(images_folder, f)) and not f.startswith('.')],
                        key=lambda f: f.lower())
    for category in categories:
        videos = sorted([f for f in os.listdir(os.path.join(images_folder, category))
                         if os.path.isdir(os.path.join(images_folder, category, f)) and not
                         f.startswith('.')], key=lambda f: f.lower())

        for video in videos:
            images = sorted([f for f in os.listdir(os.path.join(images_folder, category, video))
                                  if os.path.isfile(os.path.join(images_folder, category, video, f)) and not
                                  f.startswith('.')], key=lambda f: f.lower())
            img_paths += [[os.path.join(category, video, img), category] for img in images]
    img_paths = np.array(img_paths)
    img_paths = img_paths[split_indices, :]

    classes = img_paths[:, -1]
    encoder = LabelEncoder()
    encoder.fit(classes)
    classes = encoder.transform(classes)

    img_paths = img_paths[:, 0]

    n_instances = len(classes)
    idx = 0
    while True:
        x = []
        labels = []

        while len(labels) < batch_size:
            if idx >= n_instances:
                idx = 0
            img = image.load_img(os.path.join(images_folder, img_paths[idx]), target_size=(224, 224))
            coded_image = image.img_to_array(img)
            coded_image = np.expand_dims(coded_image, axis=0)
            coded_image = utils.preprocess_input(coded_image, version=1)  # version=1 (VGG16) or version=2 (RESNET50)
            x.append(coded_image)
            labels.append(classes[idx])
            idx += 1

        yield (np.concatenate(tuple(x), axis=0), np.array(labels))


def extract_faces_from_videos(database_folder, output_folder):

    owd = os.getcwd()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    categories = sorted([f for f in os.listdir(database_folder)
                         if os.path.isdir(os.path.join(database_folder, f)) and not f.startswith('.')],
                        key=lambda f: f.lower())

    for category in categories:

        videos = sorted([f for f in os.listdir(os.path.join(database_folder, category))
                         if os.path.isfile(os.path.join(database_folder, category, f)) and not
                         f.startswith('.')], key=lambda f: f.lower())

        input_list = " ".join(['-f "%s/%s/%s"' % (database_folder, category, video) for video in videos])

        os.chdir(config.get_openface_folder())
        command = 'build/bin/FeatureExtraction %s -out_dir "%s" -q -simalign' % \
                  (input_list, os.path.join(output_folder, category))
        subprocess.call(command, shell=True)
        os.chdir(owd)

        trash_files = sorted([f for f in os.listdir(os.path.join(output_folder, category))
                         if os.path.isfile(os.path.join(output_folder, category, f)) and not
                         f.startswith('.')], key=lambda f: f.lower())
        for trash_file in trash_files:
            os.remove(os.path.join(output_folder, category, trash_file))

        face_folders = sorted([f for f in os.listdir(os.path.join(output_folder, category))
                             if os.path.isdir(os.path.join(output_folder, category, f)) and not f.startswith('.')],
                            key=lambda f: f.lower())
        for face_folder in face_folders:
            os.rename(os.path.join(output_folder, category, face_folder),
                      os.path.join(output_folder, category, face_folder.replace("_aligned", "")))

    print("OpenFace analysis complete.")


def extract_vgg_features(images_folder, output_folder=None):

    if output_folder == None:
        output_folder = os.path.join(images_folder + "_", "vgg_features")

    # Convolution Features
    vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max

    categories = sorted([f for f in os.listdir(images_folder)
                         if os.path.isdir(os.path.join(images_folder, f)) and not f.startswith('.')],
                        key=lambda f: f.lower())

    for category in categories:

        if not os.path.exists(os.path.join(output_folder, category)):
            os.makedirs(os.path.join(output_folder, category))

        videos = sorted([f for f in os.listdir(os.path.join(images_folder, category))
                         if os.path.isdir(os.path.join(images_folder, category, f)) and not
                         f.startswith('.')], key=lambda f: f.lower())

        for video in videos:
            images = sorted([f for f in os.listdir(os.path.join(images_folder, category, video))
                                  if os.path.isfile(os.path.join(images_folder, category, video, f)) and not
                                  f.startswith('.')], key=lambda f: f.lower())

            features = []
            for i in images:
                img_path = os.path.join(images_folder, category, video, i)
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_input(x, version=1)  # version=1 (VGG16) or version=2 (RESNET50)
                features.append(vgg_features.predict(x))
                print("Predicted %s" % img_path)
            features = np.concatenate(tuple(features), axis=0)
            with open(os.path.join(output_folder, category, video + ".csv"), "w+") as write_file:
                writer = csv.writer(write_file)
                writer.writerows(features)
            a = 0


def extract_vgg_layers(images_folder, output_folder=None, layers=["fc6", "fc7"]):

    if output_folder is None:
        output_folder = os.path.join(images_folder + "_", "vgg_%s" % "_".join(layers))

    # Convolution Features
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))  # pooling: None, avg or max
    model_outputs = [Model(vgg_model.input, vgg_model.get_layer(layer_name).output) for layer_name in layers]

    categories = sorted([f for f in os.listdir(images_folder) if
                         os.path.isdir(os.path.join(images_folder, f)) and not f.startswith('.')],
                        key=lambda f: f.lower())

    for category in categories:

        if not os.path.exists(os.path.join(output_folder, category)):
            os.makedirs(os.path.join(output_folder, category))

        videos = sorted([f for f in os.listdir(os.path.join(images_folder, category))
                         if os.path.isdir(os.path.join(images_folder, category, f)) and not
                         f.startswith('.')], key=lambda f: f.lower())

        for video in videos:
            images = sorted([f for f in os.listdir(os.path.join(images_folder, category, video))
                                  if os.path.isfile(os.path.join(images_folder, category, video, f)) and not
                                  f.startswith('.')], key=lambda f: f.lower())

            features = []
            for i in images:
                img_path = os.path.join(images_folder, category, video, i)
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_input(x, version=1)  # version=1 (VGG16) or version=2 (RESNET50)
                layer_outputs = [model_output.predict(x) for model_output in model_outputs]
                features.append(np.concatenate(tuple(layer_outputs), axis=-1))
                print("Predicted %s" % img_path)
            features = np.concatenate(tuple(features), axis=0)
            with open(os.path.join(output_folder, category, video + ".csv"), "w+") as write_file:
                writer = csv.writer(write_file)
                writer.writerows(features)
            a = 0


def get_statistics(features_folder, output_path=None, statistics=["mean"]):

    if output_path is None:
        output_path = os.path.join(features_folder, os.path.split(features_folder)[1])
    features = os.path.split(features_folder)[1]
    analyzed_files = []
    matrix = []

    classes = sorted([f for f in os.listdir(features_folder)
                      if os.path.isdir(os.path.join(features_folder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    
    for class_name in classes:
        files = sorted([f for f in os.listdir(os.path.join(features_folder, class_name))
                        if os.path.isfile(os.path.join(features_folder, class_name, f)) and not f.startswith('.')
                        and f[-4:].lower() == ".csv"], key=lambda f: f.lower())
        analyzed_files += ["%s,%s" % (file, class_name) for file in files]
        for feat_file in files:
            df = pandas.read_csv(os.path.join(features_folder, class_name, feat_file), header=None)
            feature_names = df.columns.values
            feature_names = ["%s_%s" % (features, num) for num in feature_names]
            vals = df.values
            header = []
            data = []
            for statistic in statistics:
                if statistic == "max":
                    values = np.nanmax(vals, axis=0)
                elif statistic == "min":
                    values = np.nanmin(vals, axis=0)
                elif statistic == "mean":
                    values = np.nanmean(vals, axis=0)
                elif statistic == "median":
                    values = np.nanmedian(vals, axis=0)
                elif statistic == "std":
                    values = np.nanstd(vals, axis=0)
                elif statistic == "var":
                    values = np.nanvar(vals, axis=0)
                elif statistic == "kurt":
                    values = scipy.stats.kurtosis(vals, axis=0)
                elif statistic == "skew":
                    values = scipy.stats.skew(vals, axis=0)
                elif statistic == 'percentile25':
                    values = np.nanpercentile(vals, 25, axis=0)
                elif statistic == 'percentile50':
                    values = np.nanpercentile(vals, 50, axis=0)
                elif statistic == 'percentile75':
                    values = np.nanpercentile(vals, 75, axis=0)

                header += ["%s_%s" % (name, statistic) for name in feature_names]
                data.append(values)
            instance = np.concatenate(tuple(data), axis=-1).tolist()
            instance.append(class_name)
            matrix.append(instance)
            print("%s analyzed." % feat_file)
    header.append("Class")
    matrix = [header] + matrix
    am.create_arff(matrix, classes, os.path.split(output_path)[0], os.path.split(output_path)[1] + "_%s" % "_".join(statistics),
                   os.path.split(output_path)[1] + "_statistics")
    print("Statistics from %s obtained." % os.path.split(output_path)[1])
    with open(output_path + ".txt", "w+") as files:
        files.write("\n".join(analyzed_files))


def split_data(images_folder, split, seed=0):
    if split[0] + split[1] != 1.0:
        raise ValueError("Splits don't sum 1.")

    img_paths = []
    categories = sorted([f for f in os.listdir(images_folder)
                         if os.path.isdir(os.path.join(images_folder, f)) and not f.startswith('.')],
                        key=lambda f: f.lower())
    for category in categories:
        videos = sorted([f for f in os.listdir(os.path.join(images_folder, category))
                         if os.path.isdir(os.path.join(images_folder, category, f)) and not
                         f.startswith('.')], key=lambda f: f.lower())

        for video in videos:
            images = sorted([f for f in os.listdir(os.path.join(images_folder, category, video))
                             if os.path.isfile(os.path.join(images_folder, category, video, f)) and not
                             f.startswith('.')], key=lambda f: f.lower())
            img_paths += [1 for img in images]

    n_instances = len(img_paths)
    instance_indices = np.arange(n_instances)
    np.random.seed(seed)
    np.random.shuffle(instance_indices)

    n_training_instances = ceil(split[0] * n_instances)
    training_indices = instance_indices[:n_training_instances]

    validation_indices= instance_indices[n_training_instances:]
    n_validation_instances = len(validation_indices)
    return (training_indices, validation_indices), n_training_instances, n_validation_instances


def vgg_fine_tuning(images_folder, batch_size=16, epochs=10, verbose=1):
    # custom parameters
    nb_class = 1
    hidden_dim = 512
    split = (0.7, 0.3)

    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    out = Dense(nb_class, activation='softmax', name='fc8')(x)
    custom_vgg_model = Model(vgg_model.input, out)
    custom_vgg_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    splits, n_training_instances, n_validation_instances = split_data(images_folder, split, seed=1)
    train_data_generator = data_generator(images_folder, batch_size, splits[0], mode="training")
    validation_data_generator = data_generator(images_folder, batch_size, splits[1], mode="validation")
    custom_vgg_model.fit_generator(generator=train_data_generator, steps_per_epoch=ceil(n_training_instances/batch_size),
                                   epochs=epochs, verbose=verbose,
                                   validation_data=validation_data_generator,
                                   validation_steps=ceil(n_validation_instances / batch_size))

    # serialize model to JSON
    model_json = custom_vgg_model.to_json()
    with open("custom_vgg_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    custom_vgg_model.save_weights("custom_vgg_model.h5")
    print("Saved model to disk")


def load_tuned_vgg(model_path, weights_path):
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")
    return loaded_model
