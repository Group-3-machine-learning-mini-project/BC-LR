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

    # infile = "data_banknote_authentication.txt"
    # feature_list = ["variance","skewness","curtosis","entropy"]
    
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = split_data(infile, feature_list)
	# Read dataloader
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for data, label in train_dataset:
        X_train.append(parsing_data(data))
        Y_train.append(label)
    for data, label in test_dataset:
        X_test.append(parsing_data(data))
        Y_test.append(label)
    # Convert to numpy array
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    # Train SVM model with train data
    clf = svm.SVC(kernel = "rbf", C = 3, gamma = 0.3)
    clf.fit(X_train, Y_train)
    # Test trained svm model and print accuracy
    error = clf.predict(X_test) - Y_test
    accuracy = len(error[error == 0])/len(error)
    print("Accuracy of the SVM model: ",accuracy)
