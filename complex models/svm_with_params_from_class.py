import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import json
from pathlib import Path
import csv

from data.grab_and_partition import win_amounts, get_data, prob_of_error


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





def main():
    print("SVM: Converting 1s and -1s for heroes to Win Rates")
    dota_train_df, dota_test_df = get_data()

    X_train = dota_train_df.iloc[:, 1:].to_numpy()
    y_train = dota_train_df.iloc[:, 0].to_numpy()
    X_test = dota_test_df.iloc[:, 1:].to_numpy()
    y_test = dota_test_df.iloc[:, 0].to_numpy()

    X_train_encoded = encode_with_winrate(X_train)
    X_test_encoded = encode_with_winrate(X_test)

    print("Win amounts for training set")
    win_amounts(y_train)
    print("\nWin amounts for test set")
    win_amounts(y_test)

    rbf_svc = make_pipeline(StandardScaler(), SVC(kernel='rbf',  gamma=10, C=10))
    rbf_svc.fit(X_train_encoded, y_train)

    predictions_for_pipeline = rbf_svc.predict(X_test_encoded)
    print("alleged best values prob of Error", prob_of_error(predictions_for_pipeline, y_test))

    """
    rbf_svc = make_pipeline(StandardScaler(), SVC(kernel='rbf',  gamma=0.7, C=1.0))
    alleged best values prob of Error 0.4655138915873325
    
    rbf_svc = make_pipeline(StandardScaler(), SVC(kernel='rbf',  gamma=0.13103, C=0.1931))
    alleged best values prob of Error 0.4655138915873325
    """

if __name__ == '__main__':
    main()
