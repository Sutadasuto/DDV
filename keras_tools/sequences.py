from keras.preprocessing import sequence
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder

import multiprocessing
import os
import pandas


import numpy as np


def get_input_sequences(input_data, padding="avg"):
    classes = sorted([f for f in os.listdir(input_data)
                      if os.path.isdir(os.path.join(input_data, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    X = []
    Y = []
    sequence_lengths = []
    for class_name in classes:
        files = sorted([f for f in os.listdir(os.path.join(input_data, class_name))
                        if os.path.isfile(os.path.join(input_data, class_name, f)) and not f.startswith('.')
                        and f.endswith(".csv")], key=lambda f: f.lower())
        for file in files:
            df = pandas.read_csv(os.path.join(input_data, class_name, file))
            values = df.values
            nan_inds = np.where(np.isnan(values))
            values[nan_inds] = 0
            inf_inds = np.where(np.isinf(values))
            values[inf_inds] = np.sign(values[inf_inds])
            X.append(values)
            sequence_lengths.append(len(df.values))
            Y.append(class_name)
    sequence_lengths = np.array(sequence_lengths)
    X = np.array(X)
    if padding == "avg":
        length = int(sequence_lengths.mean())
    elif padding == "max":
        length = max(sequence_lengths)
    elif padding == "min":
        length = min(sequence_lengths)
    else:
        print("No padding selected, returning raw sequences.")
        length = None
    if length is not None:
        X = sequence.pad_sequences(X, maxlen=length, dtype="float64")
    Y = np.array(Y)
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    return X, Y


def kmeans_frame_selection(frames, k=20, seed=0):

    kmeans = KMeans(n_clusters=k, random_state=seed, n_jobs=int(multiprocessing.cpu_count()/2)).fit(frames)
    centers = kmeans.cluster_centers_
    frame_numbers = []
    for center in centers:
        distances = [distance[0] for distance in euclidean_distances(frames, [center])]
        frame_number = distances.index(min(distances))
        frame_numbers.append(frame_number)
    frame_numbers.sort()

    representative_frames = []
    for i in range(len(frame_numbers)):
        frame = frames[frame_numbers[i]]
        representative_frames.append(frame)

    return np.array(representative_frames), np.array(frame_numbers)


def kmeans_seq_reduction(dataset, k=20, seed=0):

    for stream_idx, stream in enumerate(dataset):
        new_dataset = []
        for instance in stream:
            instance_length = instance.shape[0]
            new_instance, frames = kmeans_frame_selection(instance, min(k, instance_length), seed)
            new_dataset.append(new_instance)
        dataset[stream_idx] = sequence.pad_sequences(new_dataset, maxlen=k, dtype="float64")
    return dataset


def kmeans_sync_seq_reduction(dataset, k=20, seed=0):

    for instance_idx in range(dataset[0].shape[0]):
        view_vectors = []
        for stream in dataset:
            view_vectors.append(stream[instance_idx])
        instance = np.concatenate(tuple(view_vectors), axis=1)
        conc_instance, frames = kmeans_frame_selection(instance, k, seed)

        for stream_idx in range(len(dataset)):
            dataset[stream_idx][instance_idx] = dataset[stream_idx][instance_idx][frames]
    for idx, stream in enumerate(dataset):
        dataset[idx] = sequence.pad_sequences(stream.tolist(), maxlen=k, dtype="float64")
    return dataset


def multiple_sequence_padding(input_streams, padding="max"):

    lengths = []
    for stream in input_streams:
        length = stream.shape[-2]
        lengths.append(length)
    lengths = np.array(lengths)

    if padding == "avg":
        max_length = int(lengths.mean())
    elif padding == "max":
        max_length = max(lengths)
    elif padding == "min":
        max_length = min(lengths)
    else:
        raise ValueError("padding must be either avg, max or min")

    for idx, x in enumerate(input_streams):
        input_streams[idx] = sequence.pad_sequences(x, maxlen=max_length, dtype="float64")

    return input_streams


def multiple_sequence_padding_means(input_streams, padding="max"):

    lengths = []
    for stream in input_streams:
        for instance in stream:
            length = instance.shape[-2]
            lengths.append(length)
    lengths = np.array(lengths)

    if padding == "avg":
        max_length = int(lengths.mean())
    elif padding == "max":
        max_length = max(lengths)
    elif padding == "min":
        max_length = min(lengths)
    else:
        raise ValueError("padding must be either avg, max or min")

    for stream_idx, stream in enumerate(input_streams):
        input_streams[stream_idx] = kmeans_seq_reduction(stream, k=max_length)

    for idx, x in enumerate(input_streams):
        input_streams[idx] = sequence.pad_sequences(x, maxlen=max_length, dtype="float64")

    return input_streams


def pad_views_sequences(views_list, length):
    padded_list = []
    for view in views_list:
        view_len = view.shape[0]
        turn = 0
        while view_len > length:
            view = np.delete(view, (turn), axis=0)
            view_len = view.shape[0]
            turn = (turn + 1) * -1
        padded_list.append(view)
    return padded_list


def synchronize_views(input_streams):

    num_instances = input_streams[0].shape[0]

    for instance_idx in range(num_instances):

        seq_lengths = []
        samples = []
        for view in input_streams:
            seq_lengths.append(view[instance_idx].shape[0])
            samples.append(view[instance_idx])
        length = min(seq_lengths)
        padded_instances = pad_views_sequences(samples, length)

        for view_idx in range(len(padded_instances)):
            input_streams[view_idx][instance_idx] = padded_instances[view_idx]

    return input_streams

