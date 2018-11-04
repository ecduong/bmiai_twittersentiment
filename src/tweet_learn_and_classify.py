import pandas as pd

PATH_TO_TRAINING_DATA = "../data/train.csv"
PATH_TO_TESTING_DATA = "../data/test.csv"

# Read the CSV files - one for training and one for testing
training_data = pd.read_csv(PATH_TO_TRAINING_DATA, encoding='ISO-8859-1')
testing_data = pd.read_csv(PATH_TO_TESTING_DATA, encoding='ISO-8859-1')

# For debugging - verify data has been loaded
#print(training_data.head())
#print(testing_data.head())
