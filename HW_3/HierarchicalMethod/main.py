import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, dendrogram


def main():
    exam_scores = pd.read_csv("Data/minmax_normalized.csv")
    exam_scores = exam_scores.mean(axis=1)
    exam_scores = exam_scores.to_frame(name="average score")

    agg_clustering = AgglomerativeClustering()
    labels = agg_clustering.fit_predict(exam_scores)

    # Linkage Matrix
    Z = linkage(exam_scores, method='ward')

    # plotting dendrogram
    plt.figure(figsize=(8, 5))
    dendro = dendrogram(Z)
    plt.title('Dendrogram')
    plt.ylabel('Euclidean distance')

    plt.show()


if __name__ == "__main__":
    main()
