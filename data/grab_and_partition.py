import pandas as pd
import zipfile
from pathlib import Path

# ValueError: Multiple files found in ZIP file. Only one file per ZIP: ['dota2Train.csv', 'dota2Test.csv']
"""
Each row of the dataset is a single game with the following features (in the order in the vector):
1. Team won the game (1 or -1)
2. Cluster ID (related to location)
3. Game mode (eg All Pick)
4. Game type (eg. Ranked)
5 - end: Each element is an indicator for a hero. 
Value of 1 indicates that a player from team '1' played as that hero and '-1' for the other team. 
Hero can be selected by only one player each game. 
This means that each row has five '1' and five '-1' values.
"""
def get_data():
    path = Path(__file__).parent / "../data/dota2Dataset.zip"
    dota_train_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Train.csv'))
    dota_test_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Test.csv'))


    return dota_train_df, dota_test_df

def win_amounts(labels):
    wins_label_one = 0
    wins_label_neg_one = 0
    for x in labels:
        if x == -1:
            wins_label_neg_one += 1
        else:
            wins_label_one += 1
    label_one_percent = wins_label_one / len(labels)
    label_neg_one_percent = 1 - label_one_percent
    print("Team 1 wins = {}\npercent of wins = {}".format(wins_label_one, label_one_percent))
    print("Team -1 wins = {}\npercent of wins = {}".format(wins_label_neg_one, label_neg_one_percent))

def game_amounts(data_frame):
    #grab jason dictionary
    # add up
    # return splits
    return None

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

if __name__ == '__main__':
    main()
