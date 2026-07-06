"""
This module includes common functions used in the EDA process.

functions:
    - describe
	- category_precentage
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
