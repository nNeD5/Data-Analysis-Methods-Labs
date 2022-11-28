import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


CLUSTERS_NUMBER = 3


def get_clusters(data: list) -> KMeans:
    k_means = KMeans(n_clusters=CLUSTERS_NUMBER)
    k_means.fit(data)
    return k_means


def save_clusters(data: list) -> None:
    k_means = get_clusters(data)
    points_number_in_clusters = []
    for k, v in Counter(k_means.labels_).items():
        points_number_in_clusters.append([k, v])
    np.savetxt("Data/clusters.csv", points_number_in_clusters, fmt="% s",
               header="cluster number, points number", comments='')

    centers = k_means.cluster_centers_
    np.savetxt("Data/centers.csv", centers, fmt="% s")


def clusters_visualization(data: list):
    k_means = get_clusters(data)
    centers = k_means.cluster_centers_

    x = [data[i][0] for i in range(0, len(data))]
    y = [data[i][1] for i in range(0, len(data))]

    plt.figure(num="Clusters")
    plt.scatter(x, y, c=k_means.labels_)

    plt.xlabel("number of line")
    plt.ylabel("average exams score")

    plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9)

    plt.show()


def main():
    exam_scores = pd.read_csv("Data/minmax_normalized.csv")
    x = range(0, len(exam_scores))
    average_exam_score = pd.Series(exam_scores.mean(axis=1))
    average_exam_score = average_exam_score.tolist()
    row_number_and_average_exam_score = list(zip(x, average_exam_score))
    clusters_visualization(row_number_and_average_exam_score)
    save_clusters(row_number_and_average_exam_score)


# def clusterization(data):
#     k_means = KMeans()
#     k_means.fit(data)
#     k_means_cluster_centers = k_means.cluster_centers_
#     print(k_means_cluster_centers)
#     print(len(k_means_cluster_centers))


# def plot_elbow_method(data):
#     inertias = []
#
#     for i in range(1, len(data)):
#         kmeans = KMeans(n_clusters=i)
#         kmeans.fit(data)
#         inertias.append(kmeans.inertia_)
#     plt.plot(range(1, len(data)), inertias, marker='o')
#     plt.title("Elbow method")
#     plt.xlabel("Number of clusters")
#     plt.ylabel("Inertia")
#     plt.savefig("Elbow method")
#
#
# def main():
#     y = pd.read_csv("Data/minmax_normalized.csv")
#     x = [i for i in range(0, len(y))]
#     plot_elbow_method(y)
#     # clusterization(y)
#     # plt.scatter(x, y[score_labels[0]], c=["#1f77b4"])
#     # plt.scatter(x, y[score_labels[1]], c=["#ff7f0e"])
#     # plt.scatter(x, y[score_labels[2]], c=["#2ca02c"])


if __name__ == "__main__":
    main()
