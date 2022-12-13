from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Get data
    df = pd.read_csv('Data/cluster_data_0-2.csv')
    print(df)
    data = df[["math score", "reading score", "writing score"]]
    y = df["label"]
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data))
    pca_2 = PCA(n_components=2)
    x = pca_2.fit_transform(data_scaled)
    

    # SVM
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    clf = SVC(kernel='poly', degree=3)
    clf.fit(x_train, y_train)
    y_predictions = clf.predict(x_test)
    print("Accuracy:", accuracy_score(y_predictions, y_test))
    print(clf.get_params())

    # Plot
    h = 0.02
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


if __name__ == '__main__':
    main()
