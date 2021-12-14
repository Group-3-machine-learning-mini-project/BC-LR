# BC-LR

### Completed work
In this mini-project, we applied a SVM model to perform the binary classification process on two different datasets : Chronic Kidney Disease and Banknote Authentication. We implemented a grid search strategy to optimize the parameter sets of SVM and obtained good results within these two datasets. Especialy with non-normalized Banknote Authentication dataset, SVM model achieved an accuracy of 1.00 on this dataset, which means the data was perfectly classified.

We also applied PCA transformation to two datasets. In Chronic Kidney Disease dataset, the SVM can perform well even with a small number of components, while in Banknote Authentication dataset,each component contributed an important part within the classification process.

### Run the program
We included all the functions inside utils.py. The main pipeline was writen in main.py. To run the classification, just run main.py.

In main.py, we included the path to two datasets in the same folder. Uncomment the dataset you want to be classified and comment another one. There are also some triggers to enable some functions such as normalization, using grid search or using PCA. Set the trigger to 1 if you want to use corresponding functions, and 0 vice-versa.



