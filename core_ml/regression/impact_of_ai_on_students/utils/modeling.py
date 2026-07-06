import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve

def print_regression_evalutaion_metrics(model , X_test, y_test):
    mse = mean_squared_error(y_test, model.predict(X_test))
    rmse = root_mean_squared_error(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    r_square = r2_score(y_test, model.predict(X_test))

    metrics_df = pd.DataFrame(
        data=[mse, rmse, mae, r_square],
        index=[
            "Mean Squared Error", "Root Mean Squared Error",
            "Mean Absolute Error", "R-Square (Goodness of Fit)"
        ],
        columns=["Metrics"]
    )
    return metrics_df


def plot_learning_curve(estimator, X_train, y_train, train_sizes, cv, scoring):
    train_sizes, train_scores, eval_scores = learning_curve(
        estimator=estimator,
        X=X_train,
        y=y_train,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(train_sizes, train_scores.mean(axis=1), label="training_score")
    ax.plot(train_sizes, eval_scores.mean(axis=1), label="evaluation_score")

    ax.set(
        title="Learning Curve",
        ylabel="Score",
        xlabel="Training Sizes"
    )

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
