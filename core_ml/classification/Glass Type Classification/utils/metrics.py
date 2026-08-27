def adjusted_r2_score(
        r2_score: float, n_samples: int, n_features: int
) -> float:
    """Calculate the adjusted R-squared metric.

    :param r2_score: The calculated R-squared
    :param n_samples: Number of the samples
    :param n_features: Number of the features
    :return: Adjusted R-squared
    """
    numerator = ((1 - r2_score) * (n_samples - 1))
    denominator = n_samples - n_features - 1
    adjusted_r2_score_ = 1 - (numerator / denominator)
    return adjusted_r2_score_
