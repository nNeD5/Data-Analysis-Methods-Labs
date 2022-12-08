from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('Data/cluster_data_0-2.csv')
    # x = df[['math score', 'reading score', 'writing score']]
    df = df.to_numpy()
    x = df[:, :3]
    y = df[:, 3]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=100)
    # model = SVC(kernel='poly', degree=4)
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(accuracy_score(predictions, y_test))

    # Reduce demiseion to plot data
    # Plot
    # z = lambda x,y: (-model.intercept_[0]-model.coef_[0][0]*x -model.coef_[0][1]*y) / model.coef_[0][2]
    # tmp = np.linspace(0,1,30)
    # x_,y_ = np.meshgrid(tmp,tmp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(x[y==0,0], x[y==0,1], x[y==0,2],'ob')
    ax.plot3D(x[y==1,0], x[y==1,1], x[y==1,2],'or')
    ax.plot3D(x[y==2,0], x[y==2,1], x[y==2,2],'og')
    # ax.plot_surface(x_, y_, z(x_, y_))
    # ax.view_init(30, 60)
    plt.show()


if __name__ == '__main__':
    main()
