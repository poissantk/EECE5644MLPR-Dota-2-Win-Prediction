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
from sklearn.model_selection import KFold # Important new include

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



# ******************************* LDA *******************************
print("******************************* LDA *******************************")
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
transformed_hero_data_train = transform_hero_data(dota_train_df, dict_train)
new_x_train = np.concatenate((x_train_no_heroes_train, transformed_hero_data_train[:, None]), axis = 1)
# print(new_x_train)
new_y_train = y_train


x_test_no_heroes_test = x_test[:, 0:3]
# print(x_test_no_heroes)
dict_test = hero_win_rate(dota_test_df)
transformed_hero_data_test = transform_hero_data(dota_test_df, dict_test)
new_x_test = np.concatenate((x_test_no_heroes_test, transformed_hero_data_test[:, None]), axis = 1)
# print(new_x_test)
new_y_test = y_test



lda = LinearDiscriminantAnalysis()
X_fit = lda.fit(new_x_train, new_y_train)  # Is a fitted estimator, not actual data to project
z_train = lda.transform(new_x_train)
z_test = lda.transform(new_x_test)
w = X_fit.coef_[0]
test_preds = lda.predict(new_x_test)

print("\nTest Set Pr(Error)\nTrained on score of heros")
print(prob_of_error(test_preds, new_y_test))




lda = LinearDiscriminantAnalysis()
X_fit = lda.fit(x_train_no_heroes_train, new_y_train)  # Is a fitted estimator, not actual data to project
z_train = lda.transform(x_train_no_heroes_train)
z_test = lda.transform(x_test_no_heroes_test)
w = X_fit.coef_[0]
test_preds = lda.predict(x_test_no_heroes_test)

print("\nTest Set Pr(Error)\nTrained on training set with no hero data")
print(prob_of_error(test_preds, new_y_test))




# ******************************* Play with data set *******************************
print("******************************* MLP *******************************")



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



def neural_net(X, y, P):
    class TwoLayerMLP(nn.Module):
        # Two-layer MLP (not really a perceptron activation function...) network class
        
        def __init__(self, input_dim, hidden_dim, C):
            super(TwoLayerMLP, self).__init__()
            # Fully connected layer WX + b mapping from input_dim (n) -> hidden_layer_dim
            self.input_fc = nn.Linear(input_dim, hidden_dim)
            # Output layer again fully connected mapping from hidden_layer_dim -> outputs_dim (C)
            self.output_fc = nn.Linear(hidden_dim, C)
            
        # Don't call this function directly!! 
        # Simply pass input to model and forward(input) returns output, e.g. model(X)
        def forward(self, X):
            # X = [batch_size, input_dim (n)]
            X = self.input_fc(X)
            # Non-linear activation function, e.g. ReLU (default good choice)
            # Could also choose F.softplus(x) for smooth-ReLU, empirically worse than ReLU
            X = F.relu(X)
            # X = [batch_size, hidden_dim]
            # Connect to last layer and output 'logits'
            y = self.output_fc(X)
            return y

        
    input_dim = X.shape[1]
    n_hidden_neurons = P
    output_dim = 2

    # It's called an MLP but really it's not...
    model = TwoLayerMLP(input_dim, n_hidden_neurons, output_dim)

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

    # Convert numpy structures to PyTorch tensors, as these are the data types required by the library
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Trained model
    model = model_train(model, X_tensor, y_tensor, criterion, optimizer, num_epochs=num_epochs)
    return np.argmax(model(X_tensor).detach().numpy(), 1), model



# Polynomial degrees ("hyperparameters") to evaluate 
p_values = np.arange(1, 51, 5)
n_p_values = p_values.shape[0] # np.max(p_values)


# Number of folds for CV
K = 10

# STEP 1: Partition the dataset into K approximately-equal-sized partitions
# Shuffles data before doing the division into folds (not necessary, but a good idea)
kf = KFold(n_splits=K, shuffle=True) 

# Allocate space for CV
# No need for training loss storage too but useful comparison
mpe_train_mk = np.empty((n_p_values, K)) # Indexed by model m, data partition k

index = 0

# STEP 2: Try all polynomial orders between 1 (best line fit) and 21 (big time overfit) M=2
for p_value in p_values:
    # K-fold cross validation
    k = 0
    # NOTE that these subsets are of the TRAINING dataset
    # Imagine we don't have enough data available to afford another entirely separate validation set
    for train_indices, valid_indices in kf.split(x_train):
        # Extract the training and validation sets from the K-fold split
        X_train_k = x_train[train_indices]
        y_train_k = y_train_0_or_1[train_indices]

        # Make predictions on both the training 
        y_train_k_pred, _ = neural_net(X_train_k, y_train_k, int(p_value))

        # Record MSE as well for this model and k-fold
        mpe_train_mk[index, k] = prob_of_error(y_train_k_pred, y_train_k)

        k += 1
    index += 1
            
# STEP 3: Compute the average MSE loss for that model (based in this case on degree d)
mpe_train_m = np.mean(mpe_train_mk, axis=1) # Model average CV loss over folds

# +1 as the index starts from 0 while the degrees start from 1
optimal_p = np.argmin(mpe_train_m) + 1
print(np.min(mpe_train_m))
print("The model selected to best fit the data without overfitting is: P={}".format(optimal_p))










_, model = neural_net(x_train, y_train_0_or_1, int(optimal_p))  

X_test_tensor = torch.FloatTensor(x_test)

y_test_pred = np.argmax(model(X_test_tensor).detach().numpy(), 1)

# Record MSE as well for this model and k-fold
valid_prob_error = prob_of_error(y_test_pred, y_test_0_or_1)

print("Valid prob error: {}".format(valid_prob_error))





y_train_pred_2, model_2 = neural_net(new_x_train, y_train_0_or_1, 50)
print("\nTraining Set Pr(Error)\nTrained on training set")
print(prob_of_error(y_train_pred_2, y_train_0_or_1))

X_test_tensor = torch.FloatTensor(new_x_test) 
y_test_pred = np.argmax(model_2(X_test_tensor).detach().numpy(), 1)
# Record MSE as well for this model and k-fold
valid_prob_error_2 = prob_of_error(y_test_pred, y_test_0_or_1)
print("\nTest Set Pr(Error)\nTrained on training set")
print(valid_prob_error_2)



y_train_pred, model = neural_net(x_train_no_heroes_train, y_train_0_or_1, 50)
print("\nTraining Set Pr(Error)\nTrained on training set")
print(prob_of_error(y_train_pred, y_train_0_or_1))

X_test_tensor = torch.FloatTensor(x_test_no_heroes_test) 
y_test_pred = np.argmax(model(X_test_tensor).detach().numpy(), 1)
# Record MSE as well for this model and k-fold
valid_prob_error_3 = prob_of_error(y_test_pred, y_test_0_or_1)
print("\nTest Set Pr(Error)\nTrained on training set")
print(valid_prob_error_3)


