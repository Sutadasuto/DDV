import os
import pandas
import numpy as np
import scipy.stats
import subprocess
#import speech_recognition as sr
import tools.arff_and_matrices as am

# Videos should be separated in class folders (all videos of class A in folder "A", and so on)
def extract_features_covarep(inputFolder, outputFolder=None, sample_rate=0.01):

    if outputFolder == None:
        outputFolder = os.path.join(os.getcwd(), "covarep_features")

    classes = sorted([f for f in os.listdir(inputFolder)
                      if os.path.isdir(os.path.join(inputFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    for className in classes:
        in_dir = os.path.join(inputFolder, className)
        out_dir = os.path.join(outputFolder, className)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        command = 'matlab -nojvm -nodisplay -nosplash -r '
        command += '"cd %s, COVAREP_feature_extraction(' % ("'%s'"%(os.path.join(os.getcwd(), "audio", "covarep")))
        command += "'%s', '%s', %s"%(in_dir, out_dir, sample_rate)
        command += '), exit"'
        subprocess.call(command, shell=True)


def get_frames_per_category(database_folder, output_folder=None):

    if output_folder is None:
        output_folder = os.path.join(os.path.split(database_folder)[0], "covarep_frames")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    classes = sorted([f for f in os.listdir(database_folder)
                      if os.path.isdir(os.path.join(database_folder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    categoryDictionary = {"voice": ["f0", "vuv"],
                          "glottal_flow": ["naq", "qoq", "h1h2", "psp", "mdq", "peakslope", "rd", "creak"],
                          "mcep": ["mcep_"],
                          "hmpdm": ["hmpdm_"],
                          "hmpdd": ["hmpdd_"],
                          }
    all = []
    for key in categoryDictionary.keys():
        all += categoryDictionary[key]
    categoryDictionary["all"] = all

    for category in categoryDictionary.keys():
        category_folder = os.path.join(output_folder, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
        for className in classes:
            if not os.path.exists(os.path.join(category_folder, className)):
                os.makedirs(os.path.join(category_folder, className))
            files = sorted([f for f in os.listdir(os.path.join(database_folder, className))
                          if os.path.isfile(os.path.join(database_folder, className, f)) and not f.startswith('.')
                            and f[-4:].lower() == ".csv"], key=lambda f: f.lower())
            for feat_file in files:
                header = []
                df = pandas.read_csv(os.path.join(database_folder, className, feat_file), header='infer')
                feature_names = df.columns.values
                for feat in feature_names:
                    reference = categoryDictionary.get(category)
                    for string in reference:
                        if feat.strip().lower().startswith(string) \
                                or feat.strip().lower().endswith(string):
                            header.append(feat)
                df1 = df[header]
                df1.to_csv(os.path.join(category_folder, className, feat_file), index=False)
        print("Frames of %s acquired." % (category))


def get_statistics_covarep(databaseFolder, processedDataFolder=None, outputFileName=None, relationName=None):

    if processedDataFolder == None:
        processedDataFolder = "datasets/acousticic"
    if outputFileName== None:
        outputFileName = "all"
    if relationName == None:
        relationName = "all_acousticical"

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
            for feat in feature_names:
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
                    if np.isinf(f):
                        mm_feats.append(np.sign(f))
                    elif np.isnan(f):
                        mm_feats.append(0)
                    else:
                        mm_feats.append(f)
            if startFlag:
                matrix = [mm_names + ["Class"]]
                startFlag = False
            matrix.append(mm_feats + [className])
    am.create_arff(matrix, classes, processedDataFolder, outputFileName, relationName)
    print("Analysis of all COVAREP features acquired.")
    with open(os.path.join(processedDataFolder, outputFileName + ".txt"), "w+") as files:
        files.write("\n".join(analyzedFiles))


def get_statistics_independently(arff_file):
    matrix, labels, relation, attributes = am.arff_to_nparray(arff_file)
    classes = list(set(labels))
    labels = labels.reshape(-1, 1)
    folder, name = os.path.split(arff_file)
    if folder == "":
        folder = os.getcwd()
    stats_names = ['max', 'min', 'mean', 'median', 'std', 'var', 'kurt', 'skew', 'percentile25', 'percentile50',
                   'percentile75']

    for stat in stats_names:
        indices = []
        subname = name.replace(".arff", "_%s" % stat)
        for attribute in attributes:
            if attribute.endswith(stat):
                indices.append(attributes.index(attribute))
        submatrix = np.concatenate((matrix[:, indices], labels), axis=-1)
        subheader = np.concatenate((np.array(attributes)[indices], np.array(["Class"])), axis=-1).reshape(1,-1)
        am.create_arff(np.concatenate((subheader, submatrix), axis=0).tolist(), classes, folder, subname, subname)


def get_statistics_per_category(databaseFolder, processedDataFolder=None):

    if processedDataFolder == None:
        processedDataFolder = "datasets/acousticic"

    classes = sorted([f for f in os.listdir(databaseFolder)
                      if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())
    stats_names = ['max', 'min', 'mean', 'median', 'std', 'var', 'kurt', 'skew', 'percentile25', 'percentile50',
                   'percentile75']

    categoryDictionary = {"voice": ["f0", "vuv"],
                          "glottal_flow": ["naq", "qoq", "h1h2", "psp", "mdq", "peakslope", "rd", "creak"],
                          "mcep": ["mcep_"],
                          "hmpdm": ["hmpdm_"],
                          "hmpdd": ["hmpdd_"],
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
                                if np.isinf(f):
                                    mm_feats.append(np.sign(f))
                                elif np.isnan(f):
                                    mm_feats.append(0)
                                else:
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


def transcript_audio(lang=None, inputAudioPath=None, outputTextFolder=None, multispeaker=None):

    if lang == None:
        lang = 'English'
    if inputAudioPath == None:
        inputAudioPath = "audio.wav"
    if outputTextFolder == None:
        outputTextFolder = "output"
    if multispeaker == None:
        multispeaker = False

    try:
        with open("audio/credentials.txt") as credentials_file:
            lines = credentials_file.readlines()
            credentials = lines[0][:-1]
    except:
        print("There was a problem reading the credentials")
        raise ValueError

    language_dict = {
        'English': "en-US_NarrowbandModel",
        'Spanish': "es-ES_NarrowbandModel",
    }
    try:
        model = language_dict.get(lang)
    except:
        print("Invalid language")
        print("Available languages: " + ",".join(language_dict.keys()))

    command = 'python3 ./audio/sttClient.py -credentials ' + credentials + ' -model ' + model + \
              ' -multispeaker ' + str(multispeaker) + ' -in "' + inputAudioPath + '" -out "' + outputTextFolder + '"'
    print("Asking for transcription")
    os.system(command)
    print("Transcription done. You can retrieve the transcription with time stamps in the ./output folder")


def transcript_files(inputFolder, outputFolder, language=None, multispeaker=None):

    if language == None:
        language = 'English'

    classes = sorted([f for f in os.listdir(inputFolder)
                      if os.path.isdir(os.path.join(inputFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    for className in classes:

        audios = sorted([os.path.join(inputFolder,className,f) for f in os.listdir(os.path.join(inputFolder, className))
                         if os.path.isfile(os.path.join(inputFolder, className,f)) and not
                         f.startswith('.')], key=lambda f: f.lower())
        for audio in audios:
            transcript_audio(language, audio, os.path.join(outputFolder, className), multispeaker)
            os.rename(os.path.join(outputFolder, className, "hypotheses.txt"),
                      os.path.join(outputFolder, className, audio.split("/")[-1].split(".")[0] + ".txt"))
            os.rename(os.path.join(outputFolder, className, "timestamps.txt"),
                      os.path.join(outputFolder, className, audio.split("/")[-1].split(".")[0] + "_timestamps.csv"))
    print("All files transcripted.")


# def transcript_files_google(inputFolder, outputFolder, language=None, multispeaker=None):
#
#     if language == None:
#         language = "en-US" #Espanol mexicano: es-MX
#
#     classes = sorted([f for f in os.listdir(inputFolder)
#                       if os.path.isdir(os.path.join(inputFolder, f)) and not f.startswith('.')],
#                      key=lambda f: f.lower())
#
#     r = sr.Recognizer()
#     for className in classes:
#
#         audios = sorted([os.path.join(inputFolder,className,f) for f in os.listdir(os.path.join(inputFolder, className))
#                          if os.path.isfile(os.path.join(inputFolder, className,f)) and not
#                          f.startswith('.')], key=lambda f: f.lower())
#         for audio in audios:
#             with sr.AudioFile(audio) as source:
#                 audio = r.record(source)  # read the entire audio file
#             transcript = str(r.recognize_google(audio, language=language))


def videos_to_audio(databaseFolder, outputFolder=None):

    if outputFolder == None:
        outputFolder = "audios"

    classes = sorted([f for f in os.listdir(databaseFolder)
                               if os.path.isdir(os.path.join(databaseFolder, f)) and not f.startswith('.')],
                     key=lambda f: f.lower())

    for givenClass in classes:

        classFolder = os.path.join(databaseFolder, givenClass)
        files = sorted([f for f in os.listdir(classFolder)
                               if os.path.isfile(os.path.join(classFolder, f)) and not f.startswith('.')
                        and f[-4:].lower() == ".mp4"], key=lambda f: f.lower())

        for file in files:

            inputVideo = os.path.join(classFolder, file)
            outputAudio = os.path.join(outputFolder, givenClass)
            if not os.path.exists(outputAudio):
                os.makedirs(outputAudio)
            outputAudio = os.path.join(outputAudio, file[:-3] + "wav")

            #command = "ffmpeg -i '" + inputVideo + "' -ab 160k -ac 2 -ar 44100 -vn '" + outputAudio + "'"
            command = 'ffmpeg -i "%s" -acodec pcm_s16le -ac 1 -ar 16000 "%s"' % (inputVideo, outputAudio)
            subprocess.call(command, shell=True)
    print("Audio extraction from videos complete.")
