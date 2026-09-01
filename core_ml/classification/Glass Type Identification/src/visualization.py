import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_distribution(df):
	for feature in df.columns[:-1]:
		label = f"Mean: {df[feature].mean():.3f}\n" + \
				f"Median: {df[feature].median():.3f}\n" + \
				f"Std: {df[feature].std():.3f}\n" \
				f"Skewness: {df[feature].skew():.3f}"
		_, ax = plt.subplots(figsize=(6, 4))
		
		sns.histplot(
			data=df,
			x=feature,
			label=label,
			bins=20,
			kde=True,
		)
		plt.legend(loc="best")
		plt.tight_layout()
		plt.show()
