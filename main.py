import os
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import matplotlib.pyplot as plt
# Self-define utils
from utils import *


if __name__ == "__main__":
	# infile = "kidney_disease.csv"
	infile = "data_banknote_authentication.txt"
	feature_list = ["variance","skewness","curtosis","entropy"]

	train_loader, valid_loader, test_loader = split_data(infile, feature_list, batch_size=32)
	# Read dataloader
	for data, label in train_loader:
		pass
