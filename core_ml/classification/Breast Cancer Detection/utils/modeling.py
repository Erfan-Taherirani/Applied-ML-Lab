"""
This module contains functions used for the model training and interpretation.

functions:
- sigmoid: Computes the Sigmoid function.
- odds: Calculates the odds of a probability.
- log_odds: Calculates the log-odds of a probability.
"""
import numpy as np
import pandas as pd

# functions used for the model interpretation:
def sigmoid(z: float) -> np.ndarray:
	"""Computes the Sigmoid function.

	:param z: Is the linear predictor output.
	:return: Is the probability of the success.
	"""
	return 1 / (1 + np.exp(-z))


def odds(probability):
    """Calculates the odds of a probability.

    :param probability: The probability of the succession of the event
    :return: The odds of the probability
    """
    return probability / (1 - probability)


def log_odds(probability):
    """Calculates the log-odds of a probability.

    :param probability: The probability of the succession of the event
    :return: Ohe log-odds of the probability
    """
    return np.log(odds(probability))
