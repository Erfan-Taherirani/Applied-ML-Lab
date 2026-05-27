"""
This modules contains the most useful functions and classes needed for the 
ML projects

Author: Erfan Taherirani
Email: e.taherirani81@gmail.com
Github: Erfan-Taherirani
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def silhouette_analysis(
        X: np.ndarray, k_range: range, print_scores: bool = False,
        visualization: bool = False, kmeans_params: dict = {}
) -> tuple[int, float]:
    """Calculate the Silhoutte Score to Find the Optimal K

    This function calculates the silhouette score for the k-means clustering
    algorithm and helps you to decide the optimal k value.

    :param X: The input features array
    :param k_range: Range of possible k values
    :param print_scores: If True prints all the scores (k, Silhouette Score),
    defaults to False
    :param visualization: If True visualizes the trend of the Silhouette score
    for different k values, defaults to False
    :param kmeans_params: Custom parameters for the KMeans, defaults to {}
    :return: A tuple of best score (k, Silhouette Score)
    """
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, **kmeans_params)
        kmeans.fit(X)
        
        silhouette_score_ = silhouette_score(X, kmeans.labels_)
        scores.append((k, silhouette_score_))
        
    scores = np.array(scores)
    best_score = scores[:, 1].max()
    best_score = scores[scores[:, 1] == best_score]
    
    if print_scores:
        print("Scores:")
        print(scores)
        
    if visualization:
        ax = sns.lineplot(x=scores[:, 0], y=scores[:, 1])
        ax.set_title("Silhouette Score Trend")
        ax.set_ylabel("Silhouette Score")
        ax.set_xlabel("K")

    print("\nBest Score: (K, Silhouette Score)")
    return best_score


def elbow_method(X: np.ndarray, k_range: range, kmeans_params: dict = {}) -> None:
    """Visualize the Inertia for the Elbow Method

    :param X: The input features array
    :param k_range: Range of possible k values
    :param kmeans_params: Custom parameters for the KMeans, defaults to {}
    """
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, **kmeans_params)
        kmeans.fit(X)
        
        scores.append((k, kmeans.inertia_))

    scores = np.array(scores)
    ax = sns.lineplot(x=scores[:, 0], y=scores[:, 1])
    ax.set_title("Elbow Method Visualization")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
