import math
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
from grab_and_partition import hero_win_rate, transform_hero_data, split_data_by_lobby
from sklearn.model_selection import KFold # Important new include

from sklearn.model_selection import GridSearchCV

import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility to visualize PyTorch network and shapes
from torchsummary import summary


np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)
torch.manual_seed(7)

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






print("******************************* MLP *******************************")

def prob_of_error(predictions, true_labels):
        correct_pred_count = 0
        for x in [0, 1]:
            correct_pred_count += len(np.argwhere((predictions == x) & (true_labels == x)))
        return 1 - (correct_pred_count / len(true_labels))

def neural_net(X, y):
            
    input_dim = X.shape[1]
    output_dim = 2

    model = nn.Sequential(
        nn.Linear(input_dim, int(input_dim / 2)),
        nn.Sigmoid(),
        nn.Linear(int(input_dim / 2), int(input_dim / 4)),
        nn.Sigmoid(),
        nn.Linear(int(input_dim / 4), int(input_dim / 8)),
        nn.Sigmoid(),
        nn.Linear(int(input_dim / 8), int(input_dim / 16)),
        nn.Sigmoid(),
        nn.Linear(int(input_dim / 16), int(input_dim / 32)),
        nn.Sigmoid(),
        nn.Linear(int(input_dim / 32), output_dim)
    )

    # Convert numpy structures to PyTorch tensors, as these are the data types required by the library
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # # It's called an MLP but really it's not...
    # probs = quick_two_layer_mlp(X_tensor)

    def model_train(model, data, labels, criterion, optimizer, num_epochs=25):
        # Apparently good practice to set this "flag" too before training
        # Does things like make sure Dropout layers are active, gradients are updated, etc.
        # Probably not a big deal for our toy network, but still worth developing good practice
        model.train()
        # Optimize the neural network
        for epoch in range(num_epochs):
            # These outputs represent the model's predicted probabilities for each class. 
            outputs = model(data)
            # Criterion computes the cross entropy loss between input and target
            loss = criterion(outputs, labels)
            # Set gradient buffers to zero explicitly before backprop
            optimizer.zero_grad()
            # Backward pass to compute the gradients through the network
            loss.backward()
            # GD step update
            optimizer.step()
            
        return model


    # Stochastic GD with learning rate and momentum hyperparameters
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to 
    # the output when validating, on top of calculating the negative log-likelihood using 
    # nn.NLLLoss(), while also being more stable numerically... So don't implement from scratch
    criterion = nn.CrossEntropyLoss()
    num_epochs = 100

    # Trained model
    model = model_train(model, X_tensor, y_tensor, criterion, optimizer, num_epochs=num_epochs)
    return np.argmax(model(X_tensor).detach().numpy(), 1), model




y_train_0_or_1 = np.zeros_like(y_train)
for index in range(y_train.shape[0]):
    if y_train[index] == 1:
        y_train_0_or_1[index] = 1
    elif y_train[index] == -1:
        y_train_0_or_1[index] = 0

y_test_0_or_1 = np.zeros_like(y_test)
for index in range(y_test.shape[0]):
    if y_test[index] == 1:
        y_test_0_or_1[index] = 1
    elif y_test[index] == -1:
        y_test_0_or_1[index] = 0



_, model = neural_net(x_train, y_train_0_or_1)  

X_test_tensor = torch.FloatTensor(x_test)

y_test_pred = np.argmax(model(X_test_tensor).detach().numpy(), 1)

# Record MSE as well for this model and k-fold
valid_prob_error = prob_of_error(y_test_pred, y_test_0_or_1)

print("Valid prob error: {}".format(valid_prob_error))


