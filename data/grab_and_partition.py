import json

import pandas as pd
import zipfile
from pathlib import Path
import numpy as np
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
    path = Path(__file__).parent / "../data/lobbies.json"
    game_types = json.load(open(path))
    types = data_frame.iloc[:, 2].to_numpy() # col 3 but with zero index col 2

    for game_mode in game_types['lobbies']:
        number_of_current_mode = len(np.argwhere(types == game_mode['id']))
        print(game_mode['name'] + ":= {}".format(number_of_current_mode))
    print("total number of games:= {}".format(len(types)))
    return None

def hero_data(data_frame):
    #grab jason dictionary
    # add up
    # return splits
    path = Path(__file__).parent / "../data/heros.json"
    hero_types = json.load(open(path))
    types = data_frame.iloc[:, 4:117].to_numpy() # col 3 but with zero index col 2
    print(types.shape)
    print(types[:, 0].shape)
    total = 0

    for hero in hero_types['heroes']:
        num_on_team_1 = 0
        num_on_team_neg1 = 0
        hero_id = hero['id']
        for element in types[:, hero_id - 1]:   #element in hero's column
            if element == 1:
                num_on_team_1 += 1
            elif element == -1:
                num_on_team_neg1 +=1
        print(hero['name'] + ":= {}".format(num_on_team_1 + num_on_team_neg1) + "           {}%".format(round(((num_on_team_1 + num_on_team_neg1) / 10282.56), 2)))
        total += (num_on_team_1 + num_on_team_neg1)
    # print("total number of games:= {}".format(len(types)))
    # print(total) = 1028256
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

    print("\nTotal game mode numbers across test and train")
    game_amounts(pd.concat([dota_train_df, dota_test_df]))

    print("\nTotal game mode numbers test")
    game_amounts(dota_test_df)

    print("\nTotal game mode numbers train")
    game_amounts(dota_train_df)

    print("\nTotal heroes numbers across test and train")
    hero_data(pd.concat([dota_train_df, dota_test_df]))

if __name__ == '__main__':
    main()
