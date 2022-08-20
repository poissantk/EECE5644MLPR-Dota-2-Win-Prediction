import pandas as pd
import zipfile
from pathlib import Path


from sys import displayhook, float_info
import matplotlib.pyplot as plt # For general plotting

import numpy as np
from pandas import array

from scipy.stats import multivariate_normal # MVN not univariate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from grab_and_partition import hero_win_rate, transform_hero_data


np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title


# ******************************* Load Data *******************************
path = Path(__file__).parent / "../data/dota2Dataset.zip"
dota_train_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Train.csv'), header=None)

dota_train = dota_train_df.to_numpy()
N_train = dota_train.shape[0]
n = dota_train.shape[1]
y_train = dota_train[:, 0]
x_train = dota_train[:, 1:]
L = np.array([-1, 1])
Nl = np.array([sum(y_train == l) for l in L])
print(Nl)

dota_test_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Test.csv'), header=None)

dota_test = dota_test_df.to_numpy()
N_test = dota_test.shape[0]
y_test = dota_test[:, 0]
x_test = dota_test[:, 1:]



# ******************************* LDA *******************************

lda = LinearDiscriminantAnalysis()
X_fit = lda.fit(x_train, y_train)  # Is a fitted estimator, not actual data to project
z_train = lda.transform(x_train)
z_test = lda.transform(x_test)
w = X_fit.coef_[0]
test_preds = lda.predict(x_test)

def prob_of_error(predictions, true_labels):
        correct_pred_count = 0
        for x in [-1, 1]:
            correct_pred_count += len(np.argwhere((predictions == x) & (true_labels == x)))
        return 1 - (correct_pred_count / len(true_labels))


print("\nTest Set Pr(Error)\nTrained on full training set")
print(prob_of_error(test_preds, y_test))



# ******************************* Play with data set *******************************

x_train_no_heroes_train = x_train[:, 0:3]
# print(x_train_no_heroes)
dict_train = hero_win_rate(dota_train_df)
transformed_hero_data_train = transform_hero_data(dota_train_df, dict_train) / 5
new_x_train = np.concatenate((x_train_no_heroes_train, transformed_hero_data_train[:, None]), axis = 1)
# print(new_x_train)
new_y_train = y_train


x_test_no_heroes_test = x_test[:, 0:3]
# print(x_test_no_heroes)
dict_test = hero_win_rate(dota_test_df)
transformed_hero_data_test = transform_hero_data(dota_test_df, dict_test) / 5
new_x_test = np.concatenate((x_test_no_heroes_test, transformed_hero_data_test[:, None]), axis = 1)
# print(new_x_test)
new_y_test = y_test



lda = LinearDiscriminantAnalysis()
X_fit = lda.fit(new_x_train, new_y_train)  # Is a fitted estimator, not actual data to project
z_train = lda.transform(new_x_train)
z_test = lda.transform(new_x_test)
w = X_fit.coef_[0]
test_preds = lda.predict(new_x_test)

def prob_of_error(predictions, true_labels):
        correct_pred_count = 0
        for x in [-1, 1]:
            correct_pred_count += len(np.argwhere((predictions == x) & (true_labels == x)))
        return 1 - (correct_pred_count / len(true_labels))


print("\nTest Set Pr(Error)\nTrained on score of heros")
print(prob_of_error(test_preds, new_y_test))




lda = LinearDiscriminantAnalysis()
X_fit = lda.fit(x_train_no_heroes_train, new_y_train)  # Is a fitted estimator, not actual data to project
z_train = lda.transform(x_train_no_heroes_train)
z_test = lda.transform(x_test_no_heroes_test)
w = X_fit.coef_[0]
test_preds = lda.predict(x_test_no_heroes_test)

def prob_of_error(predictions, true_labels):
        correct_pred_count = 0
        for x in [-1, 1]:
            correct_pred_count += len(np.argwhere((predictions == x) & (true_labels == x)))
        return 1 - (correct_pred_count / len(true_labels))


print("\nTest Set Pr(Error)\nTrained on training set with no hero data")
print(prob_of_error(test_preds, new_y_test))
