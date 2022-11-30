import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, dendrogram


def main():
    exam_scores = pd.read_csv("Data/minmax_normalized.csv")
    exam_scores = exam_scores.mean(axis=1).tolist()
    x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
    exam_scores = list(zip(x, y))
    # examx_scores = list(zip(exam_scores,
    #                         [i for i in range(0, len(exam_scores))]))
    # exam_scores = exam_scores.to_frame(name="average score")

    # agg_clustering = AgglomerativeClustering()
    # labels = agg_clustering.fit_predict(exam_scores)

    # Linkage Matrix
    linkage_data = linkage(exam_scores, method='ward', metric='euclidean')

    # plotting dendrogram
    plt.figure(figsize=(8, 5))
    dendro = dendrogram(linkage_data)
    plt.title('Dendrogram')
    plt.ylabel('Euclidean distance')

    plt.show()


if __name__ == "__main__":
    main()
