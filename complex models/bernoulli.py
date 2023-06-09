import json
from pathlib import Path
import matplotlib.pyplot as plt  # For general plotting
import matplotlib.colors as mcol
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import csv

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
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification
from data.grab_and_partition import get_data, win_amounts, get_metrics_on_results
from sklearn.naive_bayes import BernoulliNB, CategoricalNB

def prob_of_error(predictions, true_labels):
    correct_pred_count = 0
    for x in [-1, 1]:
        correct_pred_count += len(np.argwhere((predictions == x) & (true_labels == x)))
    return 1 - (correct_pred_count / len(true_labels))


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

    #pipe = make_pipeline(LogisticRegression())
    mean_cv_scores = []
    alphas = np.geomspace(10**-4, 10**4, 100)
    for a in alphas:
        mean_cv_scores.append(np.mean(cross_val_score(BernoulliNB(alpha=a), cv=10, X=X_train, y=y_train)))
    optimal_alpha = alphas[np.argmax(mean_cv_scores)]
    nb_bern = BernoulliNB(alpha=optimal_alpha)
    nb_bern.fit(X_train, y_train)
    predictions = nb_bern.predict(X_test)
    print("\nTest Set Pr(Error)\nTrained on full training set")
    print("Optimal alpha:= {}".format(optimal_alpha))
    print(prob_of_error(predictions, y_test))
    get_metrics_on_results(predictions, y_test)


    from data.grab_and_partition import split_data_by_lobby

    # ******************************* Play with data set *******************************

    print("\nTrained on train Tournament | Predictions on test Tournament")
    train_np_data_by_lobby = split_data_by_lobby(dota_train_df)
    test_np_data_by_lobby = split_data_by_lobby(dota_test_df)
    train_tournament = train_np_data_by_lobby['Tournament']
    test_tournament = test_np_data_by_lobby['Tournament']

    train_tournament = np.reshape(train_tournament, (train_tournament.shape[0], train_tournament.shape[2]))
    test_tournament = np.reshape(test_tournament, (test_tournament.shape[0], test_tournament.shape[2]))

    X_tourn_train = train_tournament[:, 1:]
    y_tourn_train = train_tournament[:, 0]
    X_tourn_test = test_tournament[:, 1:]
    y_tourn_test = test_tournament[:, 0]

    mean_cv_scores = []
    alphas = np.geomspace(10**-4, 10**4, 100)
    for a in alphas:
        mean_cv_scores.append(np.mean(cross_val_score(BernoulliNB(alpha=a), cv=10, X=X_tourn_train, y=y_tourn_train)))
    optimal_alpha = alphas[np.argmax(mean_cv_scores)]
    nb_bern = BernoulliNB(alpha=optimal_alpha)
    nb_bern.fit(X_tourn_train, y_tourn_train)
    tourn_predictions = nb_bern.predict(X_tourn_test)
    print("Tournament Test Set Pr(Error)\nTrained on full tournament training set")
    print("Optimal alpha:= {}".format(optimal_alpha))
    print(prob_of_error(tourn_predictions, y_tourn_test))
    get_metrics_on_results(tourn_predictions, y_tourn_test)



    # ******************************* Play with data set *******************************

    print("\nTrained on train Tournament With 1 V 1 | Predictions on test Tournament")
    train_np_data_by_lobby = split_data_by_lobby(dota_train_df)
    test_np_data_by_lobby = split_data_by_lobby(dota_test_df)
    train_tournament = train_np_data_by_lobby['Tournament']
    test_tournament = test_np_data_by_lobby['Tournament']

    train_tournament = np.reshape(train_tournament, (train_tournament.shape[0], train_tournament.shape[2]))
    test_tournament = np.reshape(test_tournament, (test_tournament.shape[0], test_tournament.shape[2]))

    # 1 V 1
    one_on_one_train = train_np_data_by_lobby['Solo Mid 1vs1']
    one_on_one_train = np.reshape(one_on_one_train, (one_on_one_train.shape[0], one_on_one_train.shape[2]))


    one_on_one_and_tournement = np.append(train_tournament, one_on_one_train, axis=0)
    X_one_on_one_and_tournn_train = one_on_one_and_tournement[:, 1:]
    y_one_on_one_and_tourn = one_on_one_and_tournement[:, 0]
    X_tourn_test = test_tournament[:, 1:]
    y_tourn_test = test_tournament[:, 0]
    mean_cv_scores = []
    alphas = np.geomspace(10**-4, 10**4, 100)
    for a in alphas:
        mean_cv_scores.append(np.mean(cross_val_score(BernoulliNB(alpha=a), cv=10, X=X_one_on_one_and_tournn_train,
                                                      y=y_one_on_one_and_tourn)))
    optimal_alpha = alphas[np.argmax(mean_cv_scores)]
    nb_bern = BernoulliNB(alpha=optimal_alpha)
    nb_bern.fit(X_one_on_one_and_tournn_train, y_one_on_one_and_tourn)
    tourn_onevone_predictions = nb_bern.predict(X_tourn_test)
    print("Tournament Test Set Pr(Error)\nTrained on full tournament training set with 1v1 training set")
    print("Optimal alpha:= {}".format(optimal_alpha))
    print(prob_of_error(tourn_onevone_predictions, y_tourn_test))
    get_metrics_on_results(tourn_onevone_predictions, y_tourn_test)




def data_with_online_winrates():
    print("Converting 1s and -1s for heroes to Win Rates")
    dota_train_df, dota_test_df = get_data()
    X_train = dota_train_df.iloc[:, 1:].to_numpy()
    y_train = dota_train_df.iloc[:, 0].to_numpy()
    X_test = dota_test_df.iloc[:, 1:].to_numpy()
    y_test = dota_test_df.iloc[:, 0].to_numpy()

    def encode_with_winrate(X_np):

        path = Path(__file__).parent / "../data/heros.json"

        hero_types = json.load(open(path))

        winrate_dict = {}
        reader = csv.reader(open(Path(__file__).parent / "../data/winrate_from_web.csv"))
        for row in reader:
            winrate_dict[row[0]] = float(row[1])

        hero_information = hero_types['heroes']

        # get a list of winrates in the same order as the spreadsheet columns
        # prepend ones so values other than heroes do not get changed

        wr = [1.0, 1.0, 1.0]
        for i in range(1, X_np[0, 3:].shape[0] + 1):
            assigned = False
            for hero in hero_information:
                if hero['id'] == i:
                    wr.append(winrate_dict[hero['name']])
                    assigned = True
            if not assigned:
                # catches issue where hero 24 does not exist
                wr.append(1.0)

        wr = np.array(wr)
        return [row * wr for row in X_np]

    encoded_train_X = encode_with_winrate(X_train)
    encoded_test_X = encode_with_winrate(X_test)


    mean_cv_scores = []
    alphas = np.geomspace(10**-4, 10**4, 100)
    for a in alphas:
        mean_cv_scores.append(np.mean(cross_val_score(BernoulliNB(alpha=a), cv=10, X=encoded_train_X, y=y_train)))
    optimal_alpha = alphas[np.argmax(mean_cv_scores)]
    nb_bern = BernoulliNB(alpha=optimal_alpha)
    nb_bern.fit(encoded_train_X, y_train)
    predictions = nb_bern.predict(encoded_test_X)
    print("\nEncoded Test Set Pr(Error)\nTrained on full encoded training set")
    print("Optimal alpha:= {}".format(optimal_alpha))
    print("Pr(error) converted 1s and -1s to winrate:=", prob_of_error(predictions, y_test))
    get_metrics_on_results(predictions, y_test)


if __name__ == '__main__':
    main()
    data_with_online_winrates()