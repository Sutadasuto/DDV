import csv
import os
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt


def plot_classifiers_matrix(classifiersMatrix, title, targetFolder, stopKey="all_", subtitles=None, numMetrics=2):

    if subtitles == None:
        subtitles = ["Visual Modality", "Acoustical Modality", "Textual Modality", "Fusion"]
        #subtitles = ["Visual Modality", "Acoustical Modality", "Fusion"]
    elif subtitles == False:
        subtitles = []

    if not os.path.exists(os.path.join(targetFolder, "plots")):
        os.makedirs(os.path.join(targetFolder, "plots"))

    barLabels = [cell for cell in classifiersMatrix[0][1:-2*numMetrics] if not cell == ""]
    classifierDic = {
        "LinearSVC": "SVC",
        "DecisionTreeClassifier": "DT",
        "RandomForestClassifier": "RF",
        "AdaBoostClassifier": "AdaBoost",
        "GaussianNB": "NB",
        "LogisticRegression": "LR"
    }
    for idx, label in enumerate(barLabels):
        value = classifierDic.get(label)
        if value != None:
            barLabels[idx] = value
        else:
            barLabels[idx] = label
    metrics = classifiersMatrix[1][1:numMetrics+1]

    plottableData = []
    rows = []
    for row in classifiersMatrix[2:]:

        rows.append(row[:-2*numMetrics])
        if stopKey is not None:
            if stopKey in row[0].lower():
                plottableData.append(rows)
                rows = []
    plottableData.append(rows)

    for plot, data in enumerate(plottableData):
        array = np.array(data)
        for idx, metric in enumerate(metrics,1):
            df = pandas.DataFrame(
                array[:,[colNum for colNum in range(idx,len(array[0]),numMetrics)]].astype(float),
                columns=barLabels,
                index=array[:,0]
            )
            if "modality" in subtitles[plot].lower():
                ax = df.plot(kind="bar", grid=True, width=0.85, #figsize=(4.6, 2.5),
                             title="%s - %s\n%s\n" % (title, metric, subtitles[plot]))
                ax.set_xlabel("Views")
                ax.set_ylabel(metric)
                ax.legend(loc='lower left', fontsize="small", ncol=len(barLabels))
                if metric.lower() == "accuracy":
                    ax.set_ylim((20, 80))
                else:
                    ax.set_ylim((0.20, 0.80))
            elif "fusion" in subtitles[plot].lower():
                ax = df.plot(kind="barh", grid=True, width=0.85, #figsize=(6, 3.7),
                             title="%s - %s\n%s\n" % (title, metric, subtitles[plot]))
                ax.set_ylabel("Methods")
                ax.set_xlabel(metric)
                ax.legend(loc='upper left', fontsize="small")
                if metric.lower() == "accuracy":
                    ax.set_xlim((20, 80))
                else:
                    ax.set_xlim((0.20, 0.80))
            else:
                ax = df.plot(kind="bar", grid=True, width=0.85,  # figsize=(4.6, 2.5),
                             title="%s - %s\n%s\n" % (title, metric, subtitles[plot]))
                ax.set_xlabel("Views")
                ax.set_ylabel(metric)
                ax.legend(loc='lower left', fontsize="small", ncol=len(barLabels))
                if metric.lower() == "accuracy":
                    ax.set_ylim((20, 80))
                else:
                    ax.set_ylim((0.20, 0.80))
            plt.tight_layout()
            plt.savefig(os.path.join(targetFolder, "plots", "%s_%s.png" % (subtitles[plot], metric)))
            plt.close(ax.get_figure())


def plot_complementarity_matrix(complementarityMatrix, title, targetFolder, subtitles=None, numMetrics=2):

    if subtitles == None:
        subtitles = ["Coincident Failure Diversity", "Maximum Possible Accuracy"]
    elif subtitles == False:
        subtitles = []

    if not os.path.exists(os.path.join(targetFolder, "plots")):
        os.makedirs(os.path.join(targetFolder, "plots"))

    barLabels = [cell for cell in complementarityMatrix[0][1:-2*numMetrics] if not cell == ""]
    classifierDic = {
        "LinearSVC": "SVC",
        "DecisionTreeClassifier": "DT",
        "RandomForestClassifier": "RF",
        "AdaBoostClassifier": "AdaBoost",
        "GaussianNB": "NB",
        "LogisticRegression": "LR"
    }
    for idx, label in enumerate(barLabels):
        value = classifierDic.get(label)
        if value != None:
            barLabels[idx] = value
        else:
            barLabels[idx] = label
    metrics = complementarityMatrix[1][1:numMetrics+1]

    array = np.array(complementarityMatrix)
    array = array[2:,:-2*numMetrics]
    for idx, metric in enumerate(metrics,1):
        df = pandas.DataFrame(
            array[:,[colNum for colNum in range(idx,len(array[0]),numMetrics)]].astype(float),
            columns=barLabels,
            index=array[:,0]
        )
        ax = df.plot(kind="bar", grid=True, width=0.85, #figsize=(4.6, 2.5),
                     title="%s\n%s\n" % (title, subtitles[idx-1]))
        ax.set_xlabel("Features")
        ax.set_ylabel(metric)
        ax.legend(loc='lower left', fontsize="small", ncol=len(barLabels))
        plt.tight_layout()
        plt.savefig(os.path.join(targetFolder, "plots", "%s.png" % (metric)))
        plt.close(ax.get_figure())


def plot_single_results(folder, title):

    if not os.path.exists(os.path.join(folder, "plots")):
        os.makedirs(os.path.join(folder, "plots"))

    with open(os.path.join(folder, "summary.csv")) as csv_file:
        csv_reader = csv.reader(csv_file)

        lines = []
        separators = []
        for idx, row in enumerate(csv_reader, -2):
            if idx == -2:
                classifier = row[1]
            elif idx == -1:
                metric = row[1]
            else:
                lines.append(row)
            if row[0].startswith("all_"):
                separators.append(idx)
    data = np.array(lines)

    views_indices = []
    beginning = 0
    for separator in separators:
        indices = [i for i in range(beginning, separator + 1)]
        beginning = separator + 1
        views_indices.append(indices)

    sub_data = data[beginning:, :]
    X = sub_data[:, 0]
    X = np.array([x.replace("_", " ") for x in X])
    Y = sub_data[:, 1].astype(float)
    plt.barh(X, Y, align='center')
    fig = plt.gcf()
    size = fig.get_size_inches()
    fig.set_size_inches(1.1 * size[0], size[1])
    plt.title("%s\n%s" % (title, "fusion"))
    plt.ylabel("Views")
    plt.xlabel(metric)
    plt.grid(True)
    if metric.lower() == "accuracy":
        plt.xlim((20, 80))
    else:
        plt.xlim((0.20, 0.80))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "plots", "%s_%s.png" % ("fusion", metric)))
    plt.close()
    if sub_data.shape[0] == 0:
        os.remove(os.path.join(folder, "plots", "%s_%s.png" % ("fusion", metric)))

    for indices in views_indices:
        sub_data = data[indices, :]
        subtitle = data[indices[-1], 0].replace("all_", "") + " modality"
        X = sub_data[:, 0]
        X = np.array([x.replace("_", " ") for x in X])
        Y = sub_data[:, 1].astype(float)
        plt.bar(X, Y, align='center')
        plt.title("%s\n%s" % (title, subtitle))
        plt.xlabel("Views")
        plt.ylabel(metric)
        plt.grid(True)
        plt.xticks(rotation=90)
        if metric.lower() == "accuracy":
            plt.ylim((20, 80))
        else:
            plt.ylim((0.20, 0.80))
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "plots", "%s_%s.png" % (subtitle, metric)))
        fig = plt.gcf()
        size = fig.get_size_inches()
        plt.close()

    for indices in views_indices:
        sub_data = data[indices, :]
        X = sub_data[:, 0]
        X = np.array([x.replace("_", " ") for x in X])
        Y = sub_data[:, 1].astype(float)
        plt.bar(X, Y, align='center')
    fig = plt.gcf()
    fig.set_size_inches(2*size[0], size[1])
    plt.title(title)
    plt.xlabel("Views")
    plt.ylabel(metric)
    plt.grid(True)
    plt.xticks(rotation=90)
    if metric.lower() == "accuracy":
        plt.ylim((20, 80))
    else:
        plt.ylim((0.20, 0.80))
    plt.tight_layout()
    if len(views_indices) > 0:
        plt.savefig(os.path.join(folder, "plots", "%s_%s.png" % ("All views", metric)))
    plt.close()

    df = pandas.read_csv(os.path.join(folder, "complementarity.csv"))
    data = df.values
    X = df.columns.values[1:]

    # data to plot
    n_groups = len(X)
    labels = []
    values = []
    for metric in data:
        label = metric[0]
        labels.append(label)
        val = metric[1:]
        if max(val) > 1.0:
            val = val/100.0
        values.append(val)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35

    for i in range(len(values)):
        rects = plt.bar(index + i * bar_width, values[i], bar_width, label=labels[i])

    plt.ylim((0.0, 1.0))
    plt.xticks(rotation=15)
    plt.grid(True)
    plt.xlabel('Features')
    plt.ylabel('Scores')
    plt.title('%s\n%s' % (title, "complementarity scores"))
    plt.xticks(index + bar_width, X)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder, "plots", "complementarity.png"))
    plt.close()


def s3db_errors(data, title, destination, filename):

    if not os.path.exists(destination):
        os.makedirs(destination)

    plt.plot(data, marker='o', color='r')
    plt.ylim(0,1)
    plt.grid(True)
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.title(title)
    plt.savefig(os.path.join(destination, "%s.png" % (filename)))
    plt.close()


def s3db_distributions(data, views, best_views, title, destination):

    if not os.path.exists(destination):
        os.makedirs(destination)

    for idx, iteration in enumerate(data):
        best_view = best_views[idx]
        best_view = views[best_view]
        plt.stem(iteration)
        plt.ylim(0, 0.5)
        plt.xlim(0, len(iteration)-1)
        plt.grid(True)
        plt.ylabel('Weight')
        plt.xlabel('Instance')
        plt.title("%s - Iteration %s - %s"% (title, str(idx), best_view.replace(".arff", "")))
        plt.savefig(os.path.join(destination, "%s.png" % (str(idx))))
        plt.close()


def s3db_best_views(data, num_views, title, destination, filename):

    if not os.path.exists(destination):
        os.makedirs(destination)

    plt.plot(data, 'g*', ms=20.0)
    plt.ylim(0, num_views-1)
    plt.yticks([i for i in range(num_views)])
    plt.grid(True)
    plt.ylabel('Best View Index')
    plt.xlabel('Iteration')
    plt.title(title)
    plt.savefig(os.path.join(destination, "%s.png" % (filename)))
    plt.close()
