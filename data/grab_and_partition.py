import pandas as pd
import zipfile
from pathlib import Path

# ValueError: Multiple files found in ZIP file. Only one file per ZIP: ['dota2Train.csv', 'dota2Test.csv']

path = Path(__file__).parent / "../data/dota2Dataset.zip"
dota_train_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Train.csv'))
dota_test_df = pd.read_csv(zipfile.ZipFile(path).open('dota2Test.csv'))
