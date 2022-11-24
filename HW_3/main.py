import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


score_labels = ["math score", "reading score", "writing score"]

def plot(n_clusters:int, k_means_cluster_centers:list, score) -> None:
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

    print(score[score_labels[0]])

    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors):
        cluster_center = k_means_cluster_centers[k]
        plt.plot(score[score_labels[0]], score[score_labels[1]], "w", markerfacecolor=col, marker=".")
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("KMeans")
    ax.set_xticks(())
    ax.set_yticks(())

    plt.saveifg("test.png")



def main():
    score = pd.read_csv("Data/minmax_normalized.csv")

    n_clusters = 3
    k_means = KMeans(n_clusters=n_clusters, n_init=10)
    print(test)
    k_means.fit(test)
    k_means_cluster_centers = k_means.cluster_centers_
    plot(n_clusters, k_means_cluster_centers, score)


if __name__ == '__main__':
    main()
