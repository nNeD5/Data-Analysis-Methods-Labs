from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


def main():
    df = pd.read_csv('Data/cluster_data_0-2.csv')
    # x = df[['math score', 'reading score', 'writing score']]
    df = df.to_numpy()
    x = df[:, :3]
    y = df[:, 3]

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    model = SVC(kernel='poly')
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(accuracy_score(predictions, y_test))

    xx, yy = np.mgrid[0:1:200j,0:2:200j]
    print(len(xx))
    print(len(xx[0]))
    print(xx.ravel())
    
    # z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    print("==================")
    # Plot
#    fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, edgecolors="k")
#     ax.contour(
#         x,
#         y,
#         z,
#         colors=["k", "k", "k"],
#         linestyles=["--", "-", "--"],
#         levels=[-0.5, 0, 0.5],
#     )
    # plt.show()


if __name__ == '__main__':
    main()
