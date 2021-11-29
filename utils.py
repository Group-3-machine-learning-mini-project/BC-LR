# -*- coding: utf-8 -*-
"""
This is the script containing necessary functions for this mini-project

@author: Minh-Triet, Ganglin, Binh Minh
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import torch

def category2numerical(df):
    '''
    Convert category to numeric value for classification
    '''
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df 

def normalize_dataframe(df):
    '''
    Normalize the data. This is often applied with kidney dataset.
    '''
    # Choose features with int64 or float64
    num_features = []
    for features in df.select_dtypes(include=['int64','float64']):
        num_features.append(features)
    # Scaling numerical features
    minmax = MinMaxScaler()
    df[num_features] = minmax.fit_transform(df[num_features].values)
    return df

def normalize_data(X, normalization = True):
    '''
    Normalize the data. This is often applied with kidney dataset.
    '''
    # Scaling numerical features
    minmax = MinMaxScaler()
    X = minmax.fit_transform(X)
    return X, minmax

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
		nodataval = "\t?"
		df = df.replace(nodataval, np.nan)
        # Some cleaning steps must be done manually.
		df['classification'] = df['classification'].replace(to_replace={"ckd\t":"ckd","\tckd":"ckd"})
		df['cad'] = df['cad'].replace("\tno","no")
		df['dm'] = df['dm'].replace(to_replace={"\tno":"no","\tyes":"yes"," yes":"yes"})
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

        # Preprocess: convert category to numerical values
		df = category2numerical(df)

		return df

	else:
		print("This is not the dataset we were to deal with. Please check its name to make sure it matches one of these two:\n", "data_banknote_authentication.txt\n", "kidney_disease.csv\n")


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
		- train_dataset
		- valid_dataset
		- test_dataset :
	"""
	dataset = binary_classification_dataset(infile, feature_list)
	# split ratio for training, testing and validation dataset
	testset_ratio = 0.3
	valiset_ratio = 0.0 # Here we do not create the validation set so we set as 0
	testset_length = int(len(dataset)*testset_ratio)
	valiset_length = int(len(dataset)*valiset_ratio)

	# Split dataset
	left_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset,lengths=[len(dataset)-testset_length, testset_length])
	train_dataset, valid_dataset = torch.utils.data.random_split(dataset=left_dataset,lengths=[len(left_dataset)-valiset_length, valiset_length])

	# Load dataset as dataloader 
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

	return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader

def parsing_data(data):
    '''
    Convert data from dict to array
    '''
    new_data = []
    for key in data:
      new_data.append(data[key])
    return new_data

def train_data(model, X_train, Y_train, pca = False, n_components = 4):
    '''
    Train data with pca option for dimension reduction
    '''
    if pca:
        if n_components > X_train.shape[1]:
            raise ("The number of components should be small than the number of features.")
            
        pca_model = PCA(n_components = n_components)
        X_train_reduced = pca_model.fit_transform(X_train)
        model.fit(X_train_reduced, Y_train) 
        return model, pca_model
    else:
        model.fit(X_train, Y_train)
        return model, None
    
def test_data(model, X_test, Y_test, pca = True, pca_model = None):
    '''
    Test data with pca option for dimension reduction
    '''
    target_names = ['class 0', 'class 1']
    
    print("Testing result: ")
    print("Confusion matrix:")
    print("[[TP,FN], \n[FP, TN]] \n")
    if pca:
        # Test trained svm model and print the report    
        
        print(confusion_matrix(Y_test, model.predict(pca_model.transform(X_test))))
        print("\n")
        print("Classification report: \n")
        print(classification_report(Y_test, model.predict(pca_model.transform(X_test)), 
                                    target_names=target_names))

    else:
        # Test trained svm model and print the report 
        print(confusion_matrix(Y_test, model.predict(X_test)))
        print("\n")
        print("Classification report: \n")
        print(classification_report(Y_test, model.predict(X_test), 
                                    target_names=target_names))
        
def cross_validation(model, X, Y, n_splits = 4, pca = True, n_components = 2):
    
    '''
    Cross validation. The main parameter is n_splits, which indicates
    how many folds we want to divide our dataset into.
    '''
    print("Cross validation result with {} folds:".format(n_splits))
    if pca:
        # Transform data with pca and feed into the model  
        pca_model = PCA(n_components = n_components)
        data = pca_model.fit_transform(X)
        scores = cross_val_score(model, data, Y, cv=n_splits)
        print("Scores: ", scores)
        print("%0.2f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))


    else:
        # Test trained svm model and print the report 
        scores = cross_val_score(model, X, Y, cv=n_splits)
        print("Scores: ", scores)
        print("%0.2f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))
        
    print("\n")
    
