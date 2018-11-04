import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


PATH_TO_TRAINING_DATA = "../bmiai_twittersentiment/data/train.csv"
PATH_TO_TESTING_DATA = "../bmiai_twittersentiment/data/test.csv"

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


# Read the CSV files - one for training and one for testing
training_data = pd.read_csv(PATH_TO_TRAINING_DATA, encoding='ISO-8859-1')
testing_data = pd.read_csv(PATH_TO_TESTING_DATA, encoding='ISO-8859-1')

# For debugging - verify data has been loadd
#print(training_data.head())
#print(testing_data.head())

#nltk.download() # UI to prompt user to install book - we can assume this is already done

# Clean each Tweet so that each word can be analyzed
cleaned_tweets_training = []
for tweet in training_data["SentimentText"]:
    cleaned_tweets_training.append(review_to_words(tweet))

#print(cleaned_tweets_training) # should return an array of cleaned sentences/Tweets

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(cleaned_tweets_training)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

#print(train_data_features.shape)

vocab = vectorizer.get_feature_names()
#print(vocab)

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
#for tag, count in zip(vocab, dist):
#    print(count, tag)

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100, verbose=3, n_jobs=2)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, training_data["SentimentText"] )

# Clean each Tweet so that each word can be analyzed
cleaned_tweets_testing = []
for tweet in testing_data["SentimentText"]:
    cleaned_tweets_testing.append(review_to_words(tweet))


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(cleaned_tweets_testing)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":testing_data["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
