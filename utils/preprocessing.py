import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def detect_mixed_types(df: pd.DataFrame) -> dict:
	"""Detects mixed-type columns of a dataframe

	:param df: The dataframe you want to examine
	:return: A dictionary of mixed-type columns and their types
	"""
	mixed_types = {}
	for column in df.columns:
		types = df[column].apply(lambda s: type(s).__name__).value_counts()
		if len(types) > 1:
			mixed_types[column] = types.to_dict()

	return mixed_types


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Clean a numeric column and fix it's dtype.
    
    Note: The lost data converts to Nan.

	:param series: The raw column before processing.
	:return: The processed numeric column.
	"""
    numeric_series = pd.to_numeric(series, errors="coerce")

	# check the percentage of data loss
    loss_percentage = (series.dropna().size - numeric_series.count()) / series.size * 100
    print(f"The Percentage of Data Loss: %{loss_percentage:.2f}")
    
    return numeric_series


def imputation_comaparison(df: pd.DataFrame, feature_name: str) -> plt.Figure # maybe:
	# TODO: write a code that gives a data frame and the feature's name in it that we wanna impute the missing values in it using a technique
	# ... then run multiple imputaion techniques on it and returns the histrogram distribution of the feature after each imputation technique
	pass


class missing:
	# TODO: create a dashboard for the missing values a make a comprehensive
	# visualization about the data using streamlit
	@staticmethod
	def missing_value_table(df: pd.DataFrame, include_zero_missings: bool = True) -> pd.DataFrame:
		"""Return a Concise Summary About the Missing Values

		Print shape of the dataset and return a pandas data frame about the 
		missing values consists of 'missing count' and 'missing percentage'

		:param df: Input Data Frame.
		:param include_zero_missings: Indicate where the return data frame has the 
		information of the features with zero missing values or not, defaults to True
		:return missing_df: The output data frame of missing values informations
		"""
		missing_counts = df.isnull().sum()
		missing_percentage = df.isnull().sum() / df.shape[0] * 100
		print(f"Shape of the dataset: {df.shape}")
		print(f"Number of Features with Missing Values: {missing_counts[missing_counts != 0].shape[0]}")

		if include_zero_missings:
			missing_df = pd.concat(
				[missing_counts, missing_percentage],
				axis=1,
				keys=["Missing Count", "Missing Percentage"]
			)
			return missing_df.sort_values(by="Missing Percentage", ascending=False)
		else:
			missing_df = pd.concat(
				[missing_counts, missing_percentage],
				axis=1,
				keys=["Missing Count", "Missing Percentage"]
			)
			return missing_df[missing_df["Missing Count"] != 0].sort_values(
				by="Missing Percentage",
				ascending=False
			)

	@staticmethod
	def matrix(df: pd.DataFrame) -> None:
		hm = df.isnull()
		ax = sns.heatmap(
			data=hm,
			cbar=False,
			cmap="viridis",
			yticklabels=False
		)

		ax.set_title("Matrix of the Missing Values")

	@staticmethod
	def detect_suspicious_values(series: pd.Series, suspicious_values: list) -> None:
		# TODO: write a code that process a feature and returns a report about the suspicious missing values in the feature
		pass

	@staticmethod
	def categorize_missingness(df: pd.DataFrame, threshold: int = 5) -> None:
		# TODO: write a funcion that gives a dataframe and prints a report that shows each feature category using number of missing values in the feature
		pass
