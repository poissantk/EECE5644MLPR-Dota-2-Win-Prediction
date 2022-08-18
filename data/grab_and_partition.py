import pandas as pd
import zipfile
from pathlib import Path

# ValueError: Multiple files found in ZIP file. Only one file per ZIP: ['dota2Train.csv', 'dota2Test.csv']

def get_data():
    path = Path(__file__).parent / "../data/dota2Dataset.zip"
    dota_train_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Train.csv'))
    dota_test_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Test.csv'))
    return dota_train_df, dota_test_df

def main():
    dota_train_df, dota_test_df = get_data()
    print(dota_test_df.head())
    print(dota_train_df.head())
if __name__ == '__main__':
    main()
