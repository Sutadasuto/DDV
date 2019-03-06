import os
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt


def plot_classifiers_matrix(classifiersMatrix, title, targetFolder, stopKey="all_", subtitles=None, numMetrics=2):

    if subtitles == None:
        subtitles = ["Visual Modality", "Accoustical Modality", "Textual Modality", "Fusion"]
        #subtitles = ["Visual Modality", "Accoustical Modality", "Fusion"]
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
        plt.stem(iteration, color='b')
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
