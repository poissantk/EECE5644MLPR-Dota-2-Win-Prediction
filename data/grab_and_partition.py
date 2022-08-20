import json

import pandas as pd
import zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt # For general plotting
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
    dota_train_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Train.csv'), header=None)
    dota_test_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Test.csv'), header=None)


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
    path = Path(__file__).parent / "../data/lobbies.json"
    game_types = json.load(open(path))
    types = data_frame.iloc[:, 2].to_numpy() # col 3 but with zero index col 2

    for game_mode in game_types['lobbies']:
        number_of_current_mode = len(np.argwhere(types == game_mode['id']))
        print(game_mode['name'] + ":= {}".format(number_of_current_mode))
    print("total number of games:= {}".format(len(types)))
    return None

def split_data_by_lobby(data_frame):
    path = Path(__file__).parent / "../data/lobbies.json"
    game_types = json.load(open(path))
    types = data_frame.iloc[:, 2].to_numpy() # col 3 but with zero index col 2
    lobbies_and_their_data = {}
    for game_mode in game_types['lobbies']:
        # data based on label location
        indices = np.argwhere(types == game_mode['id'])
        data = data_frame.to_numpy()[indices, :]
        print(game_mode['name'] + " shape:= {}".format(data.shape))
        lobbies_and_their_data[game_mode['name']] = data
    return lobbies_and_their_data

def hero_data(data_frame):
    #grab jason dictionary
    # add up
    # return splits
    path = Path(__file__).parent / "../data/heros.json"
    hero_types = json.load(open(path))
    types = data_frame.iloc[:, 4:117].to_numpy() # col 5 but with zero index col 4
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

def hero_win_rate(data_frame):
    path = Path(__file__).parent / "../data/heros.json"
    hero_types = json.load(open(path))
    types = data_frame.iloc[:, 4:117].to_numpy() # col 3 but with zero index col 2
    win_or_lose = data_frame.iloc[:, 0].to_numpy()
    win_rate = {}
    list_win_rate = []
    print(types.shape)
    print(types[:, 0].shape)

    for hero in hero_types['heroes']:
        win = 0
        lose = 0
        hero_id = hero['id']
        for index in range(len(types[:, hero_id - 1])):   #element in hero's column
            if (types[index, hero_id - 1] == 1 & win_or_lose[index] == 1) | (types[index, hero_id - 1] == -1 & win_or_lose[index] == -1):
                win += 1
            elif types[index, hero_id - 1] != 0:
                lose += 1

        if win + lose != 0:
            print(hero['name'] + ":= {}".format(round((win / (win + lose)), 4)))
            win_rate[hero['name']] = round((win / (win + lose)), 4)
            list_win_rate.append(round((win / (win + lose)), 4))
        else:
            print("AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH {}".format(hero['name']))
            win_rate[hero['name']] = 0
            list_win_rate.append(0)

    # from https://www.geeksforgeeks.org/python-sort-python-dictionaries-by-key-or-value/
    # print(sorted(win_rate.items(), key=lambda kv: (kv[1], kv[0])))
    print(win_rate)
    print(len(list_win_rate))
    return win_rate

def transform_hero_data(data_frame, name_to_win_rate):
    path = Path(__file__).parent / "../data/heros.json"
    types = data_frame.iloc[:, 4:117].to_numpy() # col 3 but with zero index col 2
    transformed_hero_data = np.zeros(types.shape[0])
    print(name_to_win_rate)
    id_to_heros = {}

    hero_types = json.load(open(path))
    for hero in hero_types['heroes']:
        id_to_heros[hero['id']] = hero['name']
    print(id_to_heros)

    for sample_index in range(types.shape[0]):
        sample_hero_score = 0
        if sample_index == 0:
            print(types[sample_index])
        for hero_index in range(len(id_to_heros)):
            # no hero 24
            # hero index + 1 = hero id
            if types[sample_index, hero_index] == 1:
                sample_hero_score += name_to_win_rate[id_to_heros[hero_index + 1]]
                if sample_index == 0:
                    print(hero_index)
                    print(name_to_win_rate[id_to_heros[hero_index + 1]])
            elif types[sample_index, hero_index] == -1:
                sample_hero_score = sample_hero_score - name_to_win_rate[id_to_heros[hero_index + 1]]
                if sample_index == 0:
                    print(hero_index)
                    print(name_to_win_rate[id_to_heros[hero_index + 1]])
        transformed_hero_data[sample_index] = sample_hero_score
    
    return transformed_hero_data

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

    print("\nTotal heroes win rate train")
    dict = hero_win_rate(dota_train_df)

    print("\nTransforming hero data")
    transformed_hero_data = transform_hero_data(dota_test_df, dict)
    print(transformed_hero_data)

    print(min(transformed_hero_data))
    print(max(transformed_hero_data))

    print("\nSplitting data by game type")
    lobby_df_dict = split_data_by_lobby(pd.concat([dota_train_df, dota_test_df]))
    print(lobby_df_dict.keys())

if __name__ == '__main__':
    main()
