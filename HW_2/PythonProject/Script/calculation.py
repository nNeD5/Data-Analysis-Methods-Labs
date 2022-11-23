import pandas as pd
from scipy import stats


score_columns = ["math score", "reading score", "writing score"]


def calculate(data: pd.DataFrame) -> pd.DataFrame:
    mean = data.mean()
    trimmed_mean = stats.trim_mean(data, 0.1)
    median = data.median()
    variance = data.var()
    standard_deviation = data.std()
    mean_deviation = stats.median_abs_deviation(data)

    calculated = pd.DataFrame(
        {
            "subject": ["math", "riding", "writing"],
            "mean": list(mean),
            "trimmed mean": list(trimmed_mean),
            "median": list(median),
            "variance": list(variance),
            "standard deviation": list(standard_deviation),
            "mean deviation": list(mean_deviation),
            "median absolute deviation": list(mean_deviation)
        }
    )
    return calculated


def min_max_normalize(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()

    for column in score_columns:
        min_max_difference = data[column].max() - data[column].min()
        minimal = normalized[column].min()
        normalized[column] = (normalized[column] - minimal) / min_max_difference

    return normalized


def mean_normalize(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()

    for column in score_columns:
        standard_deviation = normalized[column].std()
        mean = normalized[column].mean()
        normalized[column] = (normalized[column] - mean) / standard_deviation

    return normalized


def delete_outlier(data: pd.DataFrame) -> pd.DataFrame:
    return data[(data["math score"] > 32) &
                (data["reading score"] > 30) &
                (data["writing score"] > 30)]


def main():
    # delete null data
    exmas = pd.read_csv("../Data/exams.csv")
    exmas.dropna()
    exmas = delete_outlier(exmas)
    exmas.to_csv("exam_cleared.csv", index=False)

    # save numerical data
    exmas_score = pd.read_csv("../Data/exam_cleared.csv", usecols=score_columns)
    exmas_score.to_csv("exam_score.csv")

    calculated = calculate(exmas_score)
    calculated.to_csv("calculated.csv", index=False)

    minmax_normalized = min_max_normalize(exmas_score)
    minmax_normalized.to_csv("minmax_normalized.csv", index=False)

    mean_normalized = mean_normalize(exmas_score)
    mean_normalized.to_csv("mean_normalized.csv", index=False)


if __name__ == '__main__':
    # stuff for more information in print
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", None)
    main()
