from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from collections import Counter
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def elbow_method_visualization(data, max_clusters_number: int):
    inertias = []
    inertias_range = range(1, max_clusters_number + 1)
    for i in inertias_range:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        print(f"{i}/{max_clusters_number}")
    print("Done!")
    plt.figure(num="Elbow mehtod", figsize=(10, 5))
    plt.plot(inertias_range, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.xticks(inertias_range)
    plt.show()


CLUSTERS_NUMBER = 4


def plot_clusters(df, u_labels, label, centroids):
    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.legend()
    plt.show()


def save_clusters_and_centers(centroids, label):
    with open("Data/centroids.csv", "w") as f:
        f.write('x, y\n')
        for i in range(0, len(centroids)):
            f.write(f"{centroids[i][0]},{centroids[i][1]}\n")

    cluster_points = np.unique(label, return_counts=True)
    with open("Data/kmeans_clustres.csv", "w") as f:
        f.write("cluster, number of points\n")
        for i in range(0, CLUSTERS_NUMBER):
            f.write(f"{cluster_points[0][i]},{cluster_points[1][i]}\n")


def main():
    data = pd.read_csv("Data/minmax_normalized.csv")
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data))

    pca_2 = PCA(n_components=2)
    df = pca_2.fit_transform(data_scaled)
    elbow_method_visualization(df, 30)

    kmeans = KMeans(n_clusters=CLUSTERS_NUMBER)
    label = kmeans.fit_predict(df)
    u_labels = np.unique(label)
    centroids = kmeans.cluster_centers_
    plot_clusters(df, u_labels, label, centroids)
    save_clusters_and_centers(centroids, label)


if __name__ == "__main__":
    main()
