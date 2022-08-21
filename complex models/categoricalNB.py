import json
from pathlib import Path
import numpy as np
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
from data.grab_and_partition import get_data, win_amounts
from sklearn.naive_bayes import CategoricalNB

def prob_of_error(predictions, true_labels):
    """
    ADJUSTED TO 0, 1 FOR CATEGORICAL
    """
    correct_pred_count = 0
    for x in [0, 1]:
        correct_pred_count += len(np.argwhere((predictions == x) & (true_labels == x)))
    return 1 - (correct_pred_count / len(true_labels))


def main():
    dota_train_df, dota_test_df = get_data()
    X_train = dota_train_df.iloc[:, 1:].to_numpy()
    X_train = np.array(list(map(lambda a: list(map(lambda x: 2 if x < 0 else x, a)), X_train)))

    y_train = dota_train_df.iloc[:, 0].to_numpy()
    y_train =  np.array(list(map(lambda x: 0 if x < 0 else x, y_train)))

    X_test = dota_test_df.iloc[:, 1:].to_numpy()
    X_test = np.array(list(map(lambda a: list(map(lambda x: 2 if x < 0 else x, a)), X_test)))

    y_test = dota_test_df.iloc[:, 0].to_numpy()
    y_test =  np.array(list(map(lambda x: 0 if x < 0 else x, y_test)))

    #pipe = make_pipeline(LogisticRegression())
    mean_cv_scores = []
    alphas = np.geomspace(10**-4, 10**4, 100)
    for a in alphas:
        mean_cv_scores.append(np.mean(cross_val_score(CategoricalNB(alpha=a), cv=10, X=X_train, y=y_train)))
    optimal_alpha = alphas[np.argmax(mean_cv_scores)]
    nb_bern = CategoricalNB(alpha=optimal_alpha)
    nb_bern.fit(X_train, y_train)
    predictions = nb_bern.predict(X_test)
    print("\nTest Set Pr(Error)\nTrained on full training set")
    print("Optimal alpha:= {}".format(optimal_alpha))
    print(prob_of_error(predictions, y_test))
    """
    Optimal alpha:= 21.544346900318867

    """



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
        mean_cv_scores.append(np.mean(cross_val_score(CategoricalNB(alpha=a), cv=10, X=encoded_train_X, y=y_train)))
    optimal_alpha = alphas[np.argmax(mean_cv_scores)]
    nb_bern = CategoricalNB(alpha=optimal_alpha)
    nb_bern.fit(encoded_train_X, y_train)
    predictions = nb_bern.predict(encoded_test_X)
    print("\nEncoded Test Set Pr(Error)\nTrained on full encoded training set")
    print("Optimal alpha:= {}".format(optimal_alpha))
    print("Pr(error) converted 1s and -1s to winrate:=", prob_of_error(predictions, y_test))

    """
    Test Set Pr(Error)
    Trained on full training set
    Optimal alpha:= 21.544346900318867
    0.4033417524771712

    """
if __name__ == '__main__':
    main()
    #data_with_online_winrates()