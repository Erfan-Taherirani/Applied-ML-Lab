import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot the confusion matrix

    :param y_true: True target variables of the problem.
    :param y_pred: Predicted labels from estimators.
    """
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(
        data=cm,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        linewidths=5
    )
    ax.set(
        title="Confusion Matrix",
        ylabel="True",
        xlabel="Prediction"
    )
    plt.show()
