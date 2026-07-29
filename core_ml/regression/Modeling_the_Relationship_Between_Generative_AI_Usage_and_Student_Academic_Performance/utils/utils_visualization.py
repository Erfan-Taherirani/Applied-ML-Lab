import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(data, n_rows, n_columns, features):
    pass


def plot_categories(
        data: pd.DataFrame, features: list, n_rows: int, n_cols: int,
        figsize: tuple
) -> None:
    _, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    axs = axs.ravel()
    for i, feature in enumerate(features):
        sns.countplot(
            data=data,
            y=feature,
            ax=axs[i],
            stat="percent"
        )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = np.load(file="data/raw/data.npz", allow_pickle=True)

    values = data['data']
    features = data['columns']
    df = pd.DataFrame(
        data=values,
        columns=features
    )

    plot_categories(
        data=df,
        features=["major_category", "year_of_study"],
        n_rows=1,
        n_cols=2,
        figsize=(12, 6)
)