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
dota_train_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Train.csv'))

dota_train = dota_train_df.to_numpy()
N_train = dota_train.shape[0]
n = dota_train.shape[1]
y_train = dota_train[:, 0]
x_train = dota_train[:, 1:]
L = np.array([-1, 1])
Nl = np.array([sum(y_train == l) for l in L])
print(Nl)

dota_test_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Test.csv'))

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

