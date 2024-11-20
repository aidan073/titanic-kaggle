import preprocess
import pandas as pd
import argparse
from sklearn.svm import SVC


# arg parsing for command line
parser = argparse.ArgumentParser(description="Titanic Kaggle")
parser.add_argument('train_path', type=str, help="Path to the training dataset")
parser.add_argument('test_path', type=str, help="Path to the test dataset")
args = parser.parse_args()

# obtain dataframes
train_df = preprocess.get_df(args.train_path)
test_df = preprocess.get_df(args.test_path)

# basic visualizations for train set
preprocess.visualize(train_df, "Age", "Age Distribution", "Age", "Frequency", "blue")
preprocess.visualize(train_df, "Fare", "Fare Distribution", "Fare", "Frequency", "green")

# preprocessing
labels_train, ids_test = preprocess.seperate_labels(train_df, test_df)
train_df = preprocess.feature_engineering(train_df)
test_df = preprocess.feature_engineering(test_df)
test_df = test_df.reindex(columns=train_df.columns, fill_value=0) # ensure test has same columns as train (necessary due to get_dummies)

# SVM
svm_model = SVC(kernel='rbf', C=1.0, random_state=73)
svm_model.fit(train_df, labels_train)
y_pred = svm_model.predict(test_df)

# create submission file
submission = pd.DataFrame({
    'PassengerId': ids_test,
    'Survived': y_pred
})
submission.to_csv('submission.csv', index=False)
print("Complete")
