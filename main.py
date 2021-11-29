import os
import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
# Self-define utils
from utils import *


if __name__ == "__main__":
    infile = "kidney_disease.csv"
    feature_list = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 
                    'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    pca_flag = 0 # the trigger for using pca
    normalization = 1
    
    # infile = "data_banknote_authentication.txt"
    # feature_list = ["variance","skewness","curtosis","entropy"]
    
    # Loading dataset. The train_loader, valid_loader, test_loader are avaialble
    # in case we want to adapt a neural-network based solution with pytorch framwork.
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = split_data(infile, feature_list)
	
    # Read dataloader
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for data, label in train_dataset:
        X_train.append(parsing_data(data))
        Y_train.append(label)
    for data, label in test_dataset:
        X_test.append(parsing_data(data))
        Y_test.append(label)
        
    # If the normalization trigger is on.
    if normalization:
        # Convert to numpy array and normalize with MinmaxScaler
        # This is needed for kidney dataset
        X_train, normalizer = normalize_data(np.array(X_train))
    
        # With the test data, we reuse again the normalized minmax model used on training data    
        X_test = normalizer.transform(np.array(X_test))      
    else:
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
  
    # Train SVM model with train_data function. Depending on the pca trigger,
    # pca_model can contain fitted paramenters or None type.
    clf = svm.SVC(kernel = "rbf", C = 3, gamma = 0.5)
    clf, pca_model = train_data(clf, X_train, Y_train, pca = pca_flag, n_components = 2)
    
    # Test trained svm model and print the classification report
    test_data(clf, X_test, Y_test, pca = pca_flag, pca_model = pca_model)
