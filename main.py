import preprocess
import numpy as np
from sklearn.svm import SVC

train_path = "data/train.csv"
test_path = "data/test.csv" 

# preprocessing
train_df = preprocess.get_df(train_path)
test_df = preprocess.get_df(test_path)
labels = preprocess.seperate_labels(train_df)
train_df = preprocess.feature_engineering(train_df)
print(train_df.head())
print(train_df.isnull().sum())

# X = 
# y = data.target 

# svm_model = SVC(kernel='rbf', C=1.0, random_state=73)
# svm_model.fit(X_train, y_train)
# y_pred = svm_model.predict(X_test)
