"""
Utils for machine learning, data processing, and data analysis

Features: ...
"""
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

def missing_value_table(
	df: pd.DataFrame, include_zero_missings: bool = True
) -> pd.DataFrame:
	"""Return a Concise Summary About the Missing Values

	Print shape of the dataset and return a pandas data frame about the 
	missing values consists of 'missing count' and 'missing percentage'

	:param df: Input Data Frame.
	:param include_zero_missings: Indicate where the return data frame has the 
	information of the features with zero missing values or not, defaults to True
	:param categorize_missingness_: Categorize the data by missingness, defaults to True
	:param threshold: The threshold for the categorization of the missingness, defaults to 5
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


def detect_suspicious_values(series: pd.Series, suspicious_values: list) -> None:
	# TODO: write a code that process a feature and returns a report about the suspicious missing values in the feature
	pass

def categorize_missingness(df: pd.DataFrame, threshold: float = 30) -> None:
	"""Print Features with different missing value categories

	:param df: The input data frame
	:param threshold: The threshold that separates low and high missingness
	"""
	missing_count = df.isnull().sum()
	missing_percentage = df.isnull().sum() / df.shape[0] * 100
	missing_df = pd.DataFrame(
		{
			"missing_count": missing_count,
			"missing_percentage": missing_percentage
		}
	)
	no_missing = missing_df[missing_df['missing_count'] == 0].index
	low_missing = missing_df[(missing_df['missing_count'] != 0) & (missing_df['missing_percentage'] < threshold)].index
	high_missing = missing_df[(missing_df['missing_count'] != 0) & (missing_df['missing_percentage'] > threshold)].index
	print(f"Features with no missing values: {no_missing.to_list()}")
	print(f"Features with lower than %{threshold} missing values: {low_missing.to_list()}")
	print(f"Features with higher than %{threshold} missing values: {high_missing.to_list()}")


def describe(df: pd.DataFrame) -> pd.DataFrame:
	"""Generate a summary statistics of the dataframe

	:param df: Input dataframe
	:return: Summary statistics dataframe
	"""
	numeric_dtypes = [
		np.int8, np.int16, np.int32, np.int64, # integers
		np.float16, np.float32, np.float64, np.float128 # floats
	]
	numeric_columns = [column for column in df.columns if (df[column].dtype) in numeric_dtypes]
	df = df[numeric_columns]

	describe_df = pd.DataFrame(
    	data=[
        	df.count(), df.mean(), df.median(), df.std(), df.var(), df.min(),
        	df.quantile(0.25), df.quantile(0.5), df.quantile(0.75), df.max(), df.max() - df.min()
    	],
    	columns=df.columns,
    	index=["count", "mean", "median", "std", "variance", "min", "25%", "50%", "75%", "max", "range"]
	)

	return describe_df


def category_percentage(s: pd.Series) -> pd.DataFrame:
	"""Calculate the percentages for each category in a categorical variable.

	:param s: The inptu Feauture
	:return: The percentage data frame
	"""
	percentages = s.value_counts() / s.shape[0] * 100
	percentages_df = pd.DataFrame(percentages.apply(lambda x: f"%{x: .2f}"))
	return percentages_df.reset_index().rename(axis=1, mapper={"count": "percent"})


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


def detect_potential_outliers(
	df: pd.DataFrame,
    feature: str,
    method: str = "z-score",
    z_score_threshold: float = 3
) -> pd.Index:
    """Detect potential outliers using statistical methods
    
    This function detects potential univariate outliers using one of the
	statistical methods [z-score, IQR] from a feature in a data frame and
	return their indicies.

	:param df: The input data frame
	:param feature: Feature you want to detect its potential outliers
	:param method: Statistical method used, defaults to "z-score"
	:param z_score_threshold: threshold for z-score method, defaults to 3
	:return: Indicies of the potential outliers
	"""
    if method == "z-score":
        mean_ = df[feature].mean()
        std_ = df[feature].std()
        z_score = abs((df[feature] - mean_) / std_)
        
        potential_outliers_indicies = df.loc[(z_score > z_score_threshold), feature].index
        
    if method.upper() == "IQR":
        iqr = df[feature].quantile(0.75) - df[feature].quantile(0.25)
        lower_fence = df[feature].quantile(0.25) - (1.5 * iqr)
        upper_fence = df[feature].quantile(0.75) + (1.5 * iqr)
        
        potential_outliers_indicies = df.loc[
            (df[feature] < lower_fence) | (df[feature] > upper_fence),
            feature
		].index

    return potential_outliers_indicies # potential outliers


def imputation_comaparison(df: pd.DataFrame, feature_name: str) -> plt.Figure: # maybe
	# TODO: write a code that gives a data frame and the feature's name in it that we wanna impute the missing values in it using a technique
	# ... then run multiple imputaion techniques on it and returns the histrogram distribution of the feature after each imputation technique
	pass

def potential_outliers(df: pd.Series):
	# TODO: write a code that prints the potential outliers of a feature1
	pass

class missing:
	# TODO: create a dashboard for the missing values a make a comprehensive
	# visualization about the data using streamlit
	@staticmethod
	def detect_suspicious_values(series: pd.Series, suspicious_values: list) -> None:
		# TODO: write a code that process a feature and returns a report about the suspicious missing values in the feature
		pass

	def _categorize_missingness(self, df: pd.DataFrame, threshold: float) -> None:
		"""Print Features with different missing value categories

		:param df: The input data frame
		:param threshold: The threshold that separates low and high missingness
		"""
		missing_count = df.isnull().sum()
		missing_percentage = df.isnull().sum() / df.shape[0] * 100
		missing_df = pd.DataFrame(
			{
				"missing_count": missing_count,
				"missing_percentage": missing_percentage
			}
		)
		no_missing = missing_df[missing_df['missing_count'] == 0].index
		low_missing = missing_df[(missing_df['missing_count'] != 0) & (missing_df['missing_percentage'] < threshold)].index
		high_missing = missing_df[(missing_df['missing_count'] != 0) & (missing_df['missing_percentage'] > threshold)].index
		print(f"Features with no missing values: {no_missing.to_list()}")
		print(f"Features with lower than %{threshold} missing values: {low_missing.to_list()}")
		print(f"Features with higher than %{threshold} missing values: {high_missing.to_list()}")


class Irregularities:
	@staticmethod
	def detect_potential_outliers(S: pd.Series) -> None:
		# TODO: write a code that prints the potential outliers of a feature using `z-score` and `IQR` statistical methods.
		# it's obvious that this is an uvariate outlier detection technique
		pass

def detect_outliers(s: pd.Series) -> None:
	# TODO: write a code that detects outliers based on these techniques: `Z-score`, `IQR`, `Box Plot`, `Scatter Plot`, `LOF Algorithm`, `Isolation Forest Algorithm`
	pass


def detect_data_drift(data1: np.ndarray, data2: np.ndarray, threshold: float = 0.05) -> None:
	# TODO: write a code that calculate the p-value and warn you about the potential data drift happening
	pass


class Outlier:
	@staticmethod
	def remove_outliers(s: pd.Series, std_threshold: int) -> pd.Series:
		# TODO: write a code that gives a series and removes the outliers in it using `Z-score` approach
		pass

	@staticmethod
	def transformation(s: pd.Series, approach: str = "log") -> pd.Series:
		# TODO: write a code that transforms the feature using different approaches: Log, Box Cox, Winsorization
		pass

	@staticmethod
	def capping(s: pd.Series) -> pd.Series:
		# TODO: write a code that cap the extreme values using minimum and maximum by quartile ranges
		# this method is useful when we want to reduce the impact of extreme values but we want to 
		# preserve them in our data set to preserve their insights
		pass


def detect_anomalies(df: pd.DataFrame):
	# TODO: write a code to detect anomalies using Isolation Forest algorithm
	# in future write a code that using ensemble methods of finding anomalies and
	# then uses majority voting or weighted averaging to make decision about the anomalies
	pass


class Anomaly:
	pass


class encode:
	@staticmethod
	def target_encoding():
		pass

	@staticmethod
	def smooth_target_encoding(train, test, column, target, weight=100, fillna="global_mean"):
		# TODO: write a code to do smooth target encoding and pass out train and test data
		# formula: (category_means['count'] * category_means['mean'] + (weight * global_mean)) / (category_means['count'] + weight)
		pass

	@staticmethod
	def frequency_encoding(train, test, column, target, method="percent", fillna=0):
		# "count", "percent", "log" (np.log1p())
		# handling test data: if you see a category in the test data that you
		# didn't see in the training data so put in equal to zero or the minimum
		# frequency in the categories
		pass
