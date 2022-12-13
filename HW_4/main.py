from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random


def main():
    df = pd.read_csv('Data/cluster_data_0-2.csv')
    print(df)
    data = df[["math score", "reading score", "writing score"]]
    y = df["label"]
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data))
    pca_2 = PCA(n_components=2)
    x = pca_2.fit_transform(data_scaled)
    print(typex)

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print("Accuracy:", accuracy_score(predictions, y_test))

    w = clf.coef_[0]
    b = clf.intercept_[0]
    x_visual = np.linspace(32,57)
    y_visual = -(w[0] / w[1]) * x_visual - b / w[1]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # ax.scatter(x[:, 0], x[:, 1], c=y, edgecolors="k")
    # ax.scatter(x=x_train[0], y=x_train[1], c=y, edgecolors="k")
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
    ax.plot(x_visual, y_visual)
    plt.savefig("/mnt/d/plt.png")



if __name__ == '__main__':
    main()
