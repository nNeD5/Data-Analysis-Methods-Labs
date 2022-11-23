import matplotlib.pyplot as plt
import pandas as pd


def draw_column_from_files(file_names: list, column_name: str):
    fig, axis = plt.subplots(1, 3)
    fig.suptitle(column_name)

    for file_name, ax in zip(file_names, axis):
        exams_score = pd.read_csv(file_name)
        ax.set_title(file_name.split('.')[0])
        ax.hist(exams_score[column_name])
    fig.savefig(column_name + ".jpg")


def draw_additional_plots(file_name: str, column_names: list):
    exams_score = pd.read_csv(file_name)
    fig, axis = plt.subplots(1, 3)
    fig.suptitle("Additional plots")

    for ax, column in zip(axis, column_names):
        ax.set_title(column)
        ax.violinplot(exams_score[column])
    fig.savefig("additional_plots.jpg")


def main():
    file_names = ["exam_score.csv", "mean_normalized.csv", "minmax_normalized.csv"]
    column_names = ["math score", "reading score", "writing score"]
    # for column in column_names:
    #     draw_column_from_files(file_names, column)

    draw_additional_plots(file_names[0], column_names)

    plt.show()


if __name__ == "__main__":
    main()
