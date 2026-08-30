import pandas as pd
import numpy as np

from collections import Counter


def suspicious_values(
        feature: pd.Series, method: str = "z-score"
) -> list:
    """This function will return the suspicious values in the feature.

    In this function, we will use the univariate method z-score to identify the outliers.
    The outliers are the values that are more than 3 standard deviations away from the mean.
    We will also use the interquartile range (IQR) to identify the outliers usins method argument.

    :param feature: The feature whose outliers we want to identify.
    :param method: The method to use to identify the outliers. Can be "z-score" or "IQR".
    :return: A list of the indices of the suspicious values.
    """
    if method == "z-score":
        # z-score
        z_score = (feature - feature.mean()) / feature.std()

        # identify the outliers
        outliers = z_score[np.abs(z_score) > 3].index.tolist()
        
    if method == "IQR":
        q1 = feature.quantile(0.25)
        q3 = feature.quantile(0.75)
        iqr = q3 - q1

        # identify the outliers	
        outliers = feature[(feature < q1 - 1.5 * iqr) | (feature > q3 + 1.5 * iqr)].index.tolist()
        
    return outliers


def get_most_suspicious_values(
        df: pd.DataFrame, method: str = "z-score", threshold: int = 5
) -> list:
    """This function will return the most suspicious values in the dataframe.

    In this function, we will use the univariate methods z-score or IQR to identify the outliers.
    Then, we will use the Counter class to count the most common outliers.

    :param df: The dataframe whose outliers we want to identify.
    :param method: The method to use to identify the outliers. Can be "z-score" or "IQR".
    :param threshold: The threshold to use to identify the outliers. Default is 5.
    :return: A list of the indices of the most suspicious values.
    """
    suspicious_list = []
    for feature in df.columns[:-1]:
        for suspicious_value in suspicious_values(df[feature], method):
            suspicious_list.append(suspicious_value)

    most_suspicious_values = Counter(suspicious_list).most_common()
    outliers = []
    for value in most_suspicious_values:
        if value[1] >= threshold:
            outliers.append(value[0])

    return outliers
