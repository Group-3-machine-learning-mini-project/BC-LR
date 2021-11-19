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


def clean_data(infile):
  # Input the path to the data file
  # To check type of columns, use df.dtypes | To check NaN value, use df.isna().sum(axis=0), where df = output cleaned dataset
  filename = os.path.basename(infile)
  if filename == "data_banknote_authentication.txt":
    df = pd.read_csv(filename, header=None)
    df.columns = ["variance","skewness","curtosis","entropy","class"]
    return df

  if filename == "kidney_disease.csv":
    df = pd.read_csv(filename)
    y = df["id"]
    df = df.drop(columns="id")

    # Handle "  ?" values
    nodataval = "\t?"
    df = df.replace(nodataval, np.nan)

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
    return df

  else:
    print("This is not the dataset we were to deal with. Please check its name to make sure it matches one of these two:\n", "data_banknote_authentication.txt\n", "kidney_disease.csv\n")