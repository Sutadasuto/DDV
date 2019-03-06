import os
import ntpath
import tools.arff_and_matrices as arff

def confirm_files_per_modality(processedDataFolder=None):
    if processedDataFolder == None:
        processedDataFolder = os.path.join(os.getcwd(), "datasets")

    modality_folders = sorted([f for f in os.listdir(processedDataFolder)
                               if os.path.isdir(os.path.join(processedDataFolder, f)) and not f.startswith('.')],
                              key=lambda f: f.lower())

    names = []
    for folder in modality_folders:
        names += sorted([os.path.join(folder, f) for f in os.listdir(os.path.join(processedDataFolder, folder))
                            if os.path.isfile(os.path.join(processedDataFolder, folder, f))
                            and not f.startswith('.') and f[-4:].lower() == ".txt"],
                           key=lambda f: f.lower())

    files = [open(os.path.join(processedDataFolder, name)) for name in names]

    if len(files) == 1:
        lines = files[0].readlines()
        files[0].close()
        with open(os.path.join(processedDataFolder, names[0]), "w") as f:
            for line in lines:
                f.write(line.split(".")[0] + "\n")
        os.rename(os.path.join(processedDataFolder, ntpath.split(files[0].name)[-1]),
                  os.path.join(processedDataFolder, "list_of_instances.csv"))

        return True
    elif len(files) == 0:
        print("No list of processed files.")
        return True

    lines = [files[0].readlines()]
    tempLen = len(lines[0])
    for file in files[1:]:
        lines.append(file.readlines())
        if len(lines[-1]) != tempLen:
            return False

    list_of_instances = ""
    for i in range(tempLen):
        tempFile = lines[0][i].split(".")[0]
        temp_label = lines[0][i].split(",")[1]

        for log in lines[1:]:
            if log[i].split(".")[0] != tempFile or log[i].split(",")[1] != temp_label:
                return False
        list_of_instances += "%s,%s" % (tempFile, temp_label)

    with open(os.path.join(processedDataFolder, "list_of_instances.csv"), "w+") as f:
        f.write(list_of_instances.strip())

    for file in files:
        fileName = file.name
        file.close()
        os.remove(fileName)

    return True


def clear_error_log(errorLog=None):

    if errorLog == None:
        errorLog = "errorLog.txt"

    with open(errorLog, "w") as clearFile:
        print("Error log cleaned.")


def prune_conflictive_files(processedDataFolder=None, errorLog=None):

    if errorLog == None:
        errorLog = "errorLog.txt"
    if processedDataFolder == None:
        processedDataFolder = "datasets"

    modality_folders = sorted([f for f in os.listdir(processedDataFolder)
                               if os.path.isdir(os.path.join(processedDataFolder, f)) and not f.startswith('.')],
                              key=lambda f: f.lower())

    fileNames = []
    for folder in modality_folders:
        fileNames += sorted([os.path.join(folder, f) for f in os.listdir(os.path.join(processedDataFolder, folder))
                         if os.path.isfile(os.path.join(processedDataFolder, folder, f))
                         and not f.startswith('.') and f.endswith(".arff")],
                        key=lambda f: f.lower())

    if not confirm_files_per_modality(processedDataFolder):
        print("The processed files were not the same for the given modalities!")
        raise IOError

    try:
        log = open(errorLog)
    except:
        print("No error log. No files pruned.")
        return

    with open(os.path.join(processedDataFolder, "pruned_files.csv"), "w+") as exitLog:
        print("Previous version of 'pruned_files.txt' cleaned.")

    lines = log.readlines()
    files = []

    for line in lines:
        line = line.split(", Error")[0].split("/")[-1].split(".")[0]
        if line not in files:
            files.append(line)

    indices = []
    try:
        with open(os.path.join(processedDataFolder, "list_of_instances.csv")) as log:

            lines = log.readlines()
            for file in files:
                for row in range(len(lines)):
                    if file in lines[row]:
                        indices.append(row)
                        break
    except IOError as err:
        errorMsg = "Be sure that 'list_of_instances.csv' exists inside of " + processedDataFolder
        raise err(errorMsg)
    indices.sort()

    fileLocations = []
    for fileName in fileNames:
        fileLocations.append(os.path.join(processedDataFolder, fileName))

    for file in fileLocations:
        with open(file, "r") as f:
            lines = f.readlines()

        with open(file, "w") as f:
            currentIndex = -1
            for line in lines:
                if not line.startswith("@") and not line.startswith("\n"):
                    currentIndex += 1
                if currentIndex not in indices:
                    f.write(line)

    with open(os.path.join(processedDataFolder, "list_of_instances.csv")) as log:
        lines = log.readlines()
    with open(os.path.join(processedDataFolder, "list_of_instances.csv"),"w") as log:
        for line in lines:
            if line.strip() not in files:
                log.write(line)

    for file in files:
        with open(os.path.join(processedDataFolder, "pruned_files.csv"), "a+") as exitLog:
            exitLog.write(file + "\n")
            print(file + " pruned.")