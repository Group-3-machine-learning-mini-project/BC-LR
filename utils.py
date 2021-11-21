# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 22:31:39 2021

@author: Laptop MSI
"""

import os
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import matplotlib.pyplot as plt
import torch

def clean_data(infile):
	"""
	Replace missing values by average or median values
	Center and normalize the data
	"""
	# Input the path to the data file
	# To check type of columns, use df.dtypes | To check NaN value, use df.isna().sum(axis=0), where df = output cleaned dataset
	filename = os.path.basename(infile)
	if filename == "data_banknote_authentication.txt":
		df = pd.read_csv(filename, header=None)
		df.columns = ["variance","skewness","curtosis","entropy","classification"]
		return df

	if filename == "kidney_disease.csv":
		df = pd.read_csv(filename)
		y = df["id"]
		df = df.drop(columns="id")

		# Handle "	?" values
		nodataval = "?"
		df = df.replace(nodataval, np.nan)
    # df = df.str.replace('\t',"")
    # df = df.str.replace(' ',"")

		# Convert false string columns
		other_numeric_columns = ["pcv", "wc", "rc"]
		df[other_numeric_columns] = df[other_numeric_columns].apply(pd.to_numeric)

		# Use categorical data type
		is_str_cols = df.dtypes == object
		str_columns = df.columns[is_str_cols]
		categoric_columns = pd.Index(set(str_columns) - set(other_numeric_columns))
		df[str_columns] = df[str_columns].astype("category")

		# Handle NaN values
		# For numerical columns, there are some that has discrete values, which means that they can just have some specific values, 
		# so we have to replace their NaN with their max values. They are: sg, al and su
		discrete_numerics = ["sg","al","su"]
		fillna_mean_cols = pd.Index(
				set(df.columns[df.dtypes == "float64"]) - set(discrete_numerics))
		fillna_most_cols = pd.Index(
				set(df.columns[df.dtypes == "category"]) | set(discrete_numerics))
		assert set(fillna_mean_cols.union(fillna_most_cols)) == set(df.columns)
		df[fillna_mean_cols] = df[fillna_mean_cols].fillna(df[fillna_mean_cols].mean())
		df[fillna_most_cols] = df[fillna_most_cols].fillna(df[fillna_most_cols].mode().iloc[0])

		# Further cleaning
		df['classification'] = df['classification'].replace(to_replace={"ckd\t":"ckd","\tckd":"ckd"})
		df['cad'] = df['cad'].replace("\tno","no")
		df['dm'] = df['dm'].replace(to_replace={"\tno":"no","\tyes":"yes"," yes":"yes"})
    
		return df

	else:
		print("This is not the dataset we were to deal with. Please check its name to make sure it matches one of these two:\n", "data_banknote_authentication.txt\n", "kidney_disease.csv\n")

def category2numerical(df):
  cat_columns = df.select_dtypes(['category']).columns
  df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
  return df 

# Define dataset
class binary_classification_dataset(torch.utils.data.Dataset):
	def __init__(self, infile, feature_list):
		self.dataset_name = os.path.basename(infile).split(".")[0]
		self.data_dic = clean_data(infile).to_dict('list')
		# self.features = list(self.data_dic.keys())
		self.feature_list = feature_list

	def __len__(self):
		return len(self.data_dic["classification"])

	def __getitem__(self, idx):
		data = {}		
		for feat in self.feature_list:
			data[feat] = self.data_dic[feat][idx]

		label = self.data_dic["classification"][idx]
		return data, label 

def split_data(infile, feature_list, batch_size=32):
	"""
	Split between training set and test set
	Split the training set for cross-validation
	Inputs: 
		- df: 	
			type of pandas.core.frame.DataFrame
			cleaned data of Banknote Authentication Dataset or Chronic Kidney Disease
		- batch_size:
			batch size for dataloader

	Outputs:
		- train_loader:
		- valid_loader:
		- test_loader : 
	"""
	dataset = binary_classification_dataset(infile, feature_list)
	# split ratio for training, testing and validation dataset
	testset_ratio = 0.2
	valiset_ratio = 0.1
	testset_length = int(len(dataset)*testset_ratio)
	valiset_length = int(len(dataset)*valiset_ratio)

	# Split dataset
	left_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset,lengths=[len(dataset)-testset_length, testset_length])
	train_dataset, valid_dataset = torch.utils.data.random_split(dataset=left_dataset,lengths=[len(left_dataset)-valiset_length, valiset_length])

	# Load dataset as dataloader 
	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

	return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader

def split_data_numpy(df_file, feature_list):
  """
  This function outputs numpy format
  Split between training set and test set
  Split the training set for cross-validation
  Inputs: 
    - df: 	
      type of pandas.core.frame.DataFrame
      cleaned data of Banknote Authentication Dataset or Chronic Kidney Disease

  Outputs:
    - train data:
    - valid data:
    - test data : 
  """
  dataset = df_file.to_numpy()
  # split ratio for training, testing and validation dataset
  testset_ratio = 0.2
  valiset_ratio = 0.1
  testset_length = int((dataset.shape[0])*testset_ratio)
  valiset_length = int((dataset.shape[0])*valiset_ratio)
  trainset_length = dataset.shape[0] - testset_length - valiset_length

  # Random shuffle dataset
  np.random.shuffle(dataset)
  # Split dataset
  X_train = dataset[:trainset_length, :-1]
  X_test = dataset[trainset_length:trainset_length + testset_length, :-1]
  Y_train = dataset[:trainset_length, -1]
  Y_test = dataset[trainset_length:trainset_length + testset_length, -1]
  return X_train, Y_train, X_test, Y_test



