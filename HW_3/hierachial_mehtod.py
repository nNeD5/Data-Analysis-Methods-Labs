import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

from collections import Counter
import csv


CLUSTERS_NUMBER = 4


def main():
    data = pd.read_csv("Data/minmax_normalized.csv")

    # plot dendogram
    plt.figure(figsize=(8, 5))
    linkage_data = linkage(data, method='ward', metric='euclidean')
    dwendro = dendrogram(linkage_data)
    plt.title('Dendrogram')
    plt.ylabel('Euclidean distance')
    plt.axhline(y=4, color='red', linestyle="--")

    # clusterization
    cluster = AgglomerativeClustering(
        n_clusters=CLUSTERS_NUMBER, affinity='euclidean', linkage='ward')
    labels = cluster.fit_predict(data)
    data_with_clusters = data.copy()

    labels_csv_file = "Data/hierarchy_method.csv"
    values_in_cluster = dict(Counter(labels.tolist()))
    with open(labels_csv_file, "w") as f:
        f.write("clusters, number of points\n")
        for i in range(0, CLUSTERS_NUMBER):
            f.write(f"{i}," + str(values_in_cluster[i]) + '\n')
    plt.show()


if __name__ == "__main__":
    main()
