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
    # We also include the CLEAN step in this function
    # The detail of CLEAN step can be found in utils.py file
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = split_data(infile, feature_list)
	
    # Read data
    X_train, Y_train, X_test, Y_test = read_dataset(train_dataset, valid_dataset, test_dataset)
    
    # If the normalization trigger is on.
    if normalization:
        # Convert to numpy array and normalize with MinmaxScaler
        # This is needed for kidney dataset
        X_train, normalizer = normalize_data(X_train)
    
        # With the test data, we reuse again the normalized minmax model used on training data    
        X_test = normalizer.transform(X_test)      

    # Initilize SVM model
    clf = svm.SVC(kernel = "rbf", C = 150, gamma = 0.5)
    
    # Applying cross validation to find the best parameters
    cross_validation(clf, X_train, Y_train, n_splits = 5, pca = pca_flag, n_components = 2)
    
    # Train SVM model with train_data function. Depending on the pca trigger,
    # pca_model can contain fitted paramenters or None type.
    clf, pca_model = train_data(clf, X_train, Y_train, pca = pca_flag, n_components = 2)

    # Test trained svm model and print the classification report
    test_data(clf, X_test, Y_test, pca = pca_flag, pca_model = pca_model)
