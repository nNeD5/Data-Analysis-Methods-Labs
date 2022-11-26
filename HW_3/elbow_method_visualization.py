import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def elbow_method_visualization(data: list, max_clusters_number: int):
    inertias = []
    inertias_range = range(1, max_clusters_number + 1)
    for i in inertias_range:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        print(f"{i}/{max_clusters_number}")
    plt.figure(num="Elbow mehtod", figsize=(10, 5))
    plt.plot(inertias_range, inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.xticks(inertias_range)
    plt.show()


def main():
    data = pd.read_csv("Data/minmax_normalized.csv")
    data = pd.read_csv("Data/minmax_normalized.csv")
    average_exam_score = pd.Series(data.mean(axis=1))
    elbow_method_visualization(data, 30)


if __name__ == "__main__":
    main()
