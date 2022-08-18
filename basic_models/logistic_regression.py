import matplotlib.pyplot as plt  # For general plotting
import matplotlib.colors as mcol
import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC


"""
Currently using all data
Not differentiating between game mode or competition type
May be skewing data
"""
np.random.seed(7)

# https://stackoverflow.com/questions/70220437/can-you-use-lda-linear-discriminant-analysis-as-part-of-sklearn-pipeline-for-p
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from data.grab_and_partition import get_data, win_amounts

def main():
    dota_train_df, dota_test_df = get_data()
    X_train = dota_train_df.iloc[:, 1:].to_numpy()
    y_train = dota_train_df.iloc[:, 0].to_numpy()
    X_test = dota_test_df.iloc[:, 1:].to_numpy()
    y_test = dota_test_df.iloc[:, 0].to_numpy()

    print("Win amounts for training set")
    win_amounts(y_train)
    print("\nWin amounts for test set")
    win_amounts(y_test)

    # iloc accesses rows/columns by indexing
    # Extracting data matrix X and target labels vector
    dota_train_df, dota_test_df = get_data()
    X_train = dota_train_df.iloc[:, 1:].to_numpy()
    y_train = dota_train_df.iloc[:, 0].to_numpy()
    X_test = dota_test_df.iloc[:, 1:].to_numpy()
    y_test = dota_test_df.iloc[:, 0].to_numpy()

    pipe = make_pipeline(LogisticRegression())

    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)


    def prob_of_error(predictions, true_labels):
        correct_pred_count = 0
        for x in [-1, 1]:
            correct_pred_count += len(np.argwhere((predictions == x) & (true_labels == x)))
        return 1 - (correct_pred_count / len(true_labels))


    print("\nTest Set Pr(Error)\nTrained on full training set")
    print(prob_of_error(predictions, y_test))
if __name__ == '__main__':
    main()
