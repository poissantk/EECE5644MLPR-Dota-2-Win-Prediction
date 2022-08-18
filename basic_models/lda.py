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

# print(z)
# print(y_train)




# ******************************* ROC *******************************

# Generate ROC curve samples
def estimate_roc(discriminant_score, label):
    Nlabels = np.array((sum(label == 0), sum(label == 1)))

    # Sorting necessary so the resulting FPR and TPR axes plot threshold probabilities in order as a line
    sorted_score = sorted(discriminant_score)
    print("1")

    # Use gamma values that will account for every possible classification split
    gammas = ([sorted_score[0] - float_info.epsilon] +
              sorted_score +
              [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= g for g in gammas]
    print("2")

    ind10 = [np.argwhere((d==1) & (label==0)) for d in decisions]
    p10 = [len(inds)/Nlabels[0] for inds in ind10]
    ind11 = [np.argwhere((d==1) & (label==1)) for d in decisions]
    p11 = [len(inds)/Nlabels[1] for inds in ind11]
    print("3")

    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11))
    print("4")

    return roc, gammas

# Estimate the ROC curve for this LDA classifier
roc_lda, gamma_lda = estimate_roc(z_test, y_test)

# ROC returns FPR vs TPR, but prob error needs FNR so take 1-TPR
prob_error_lda = np.array((roc_lda[0,:], 1 - roc_lda[1,:])).T.dot(Nl.T / N_test)

# Min prob error
min_prob_error_lda = np.min(prob_error_lda)
print(min_prob_error_lda)
min_ind = np.argmin(prob_error_lda)
print(min_ind)
opt_gamma_lda = gamma_lda[min_ind]
print(opt_gamma_lda)



