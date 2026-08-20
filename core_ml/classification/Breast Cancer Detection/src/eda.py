from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_breast_cancer_data():
	"""
	Load the breast cancer dataset from sklearn and return it as a pandas DataFrame.

	Returns:
		df (pd.DataFrame): A DataFrame containing the breast cancer dataset.
	"""

	# Load the dataset
	data = load_breast_cancer()
	
	# Create a DataFrame
	df = pd.DataFrame(data.data, columns=data.feature_names)
	
	# Add the target variable to the DataFrame
	df['target'] = data.target
	
	return df
