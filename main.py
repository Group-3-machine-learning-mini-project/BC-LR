import os
import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
# Self-define utils
from utils import *


if __name__ == "__main__":
    # infile = "kidney_disease.csv"
    # feature_list = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 
    #                 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    pca_flag = 0 # the trigger for using pca
    normalization = 1 # The trigger for using normalization
    grid_search = 0 # The trigger for using automated parmeter tunning process
    
    infile = "data_banknote_authentication.txt"
    feature_list = ["variance","skewness","curtosis","entropy"]
    
    # Loading dataset. The train_loader, valid_loader, test_loader are avaialble
    # in case we want to adapt a neural-network based solution with pytorch framwork.
    # We also include the CLEAN step in this function
    # The detail of CLEAN step can be found in utils.py file
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = split_data(infile, feature_list)
	
    # Read data
    X_train, Y_train, X_test, Y_test = read_dataset(train_dataset, valid_dataset, test_dataset)
    
    # If the normalization trigger is on.
    if normalization:
        # Normalize with MinmaxScaler
        # This is essential for kidney dataset
        X_train, normalizer = normalize_data(X_train)
    
        # With the test data, we reuse again the normalized minmax model used on training data    
        X_test = normalizer.transform(X_test)      
   
    if grid_search: # Automated tunning with a set of predefined params
        
        # Initilize a SVM model
        clf = svm.SVC()
        
        # Applying grid search to find the best parameters
        best_clf = model_selection(clf, X_train, Y_train, n_splits = 5, pca = pca_flag, n_components = 2, dataname = infile)
    
    else: # Manual tunning with cross validation
        best_clf = svm.SVC(kernel = "rbf", C = 300, gamma = 0.5)
        cross_validation(best_clf, X_train, Y_train, n_splits = 5, pca = pca_flag, n_components = 2)
        
    # Train SVM model with the best model. Depending on the pca trigger,
    # pca_model can contain fitted paramenters or None type.
    best_clf, pca_model = train_data(best_clf, X_train, Y_train, pca = pca_flag, n_components = 2)

    # # Test trained svm model and print the classification report
    test_data(best_clf, X_test, Y_test, pca = pca_flag, pca_model = pca_model)
