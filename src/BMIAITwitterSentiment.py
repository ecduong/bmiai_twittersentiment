#!/usr/bin/env python

import pandas as pd
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

class BMIAITwitterSentiment:
    def __init__(self):
        self.vectorizer = None
        self.forest = None

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

    # Trains a random forest bag of words, and saves a copy of the vectorizer and
    # random forest as files for future importing
    # Parameters: path_to_training_data_csv - path to CSV file for training
    # Returns: none
    def train_data( self, path_to_training_data_csv ):
        # Read the CSV file for training
        training_data = pd.read_csv(path_to_training_data_csv, encoding='ISO-8859-1')

        # For debugging - verify data has been loadd
        #print(training_data.head())
        #print(testing_data.head())

        nltk.download() # UI to prompt user to install book

        # Clean each Tweet so that each word can be analyzed
        cleaned_tweets_training = []
        for tweet in training_data["SentimentText"]:
            cleaned_tweets_training.append(self.review_to_words(tweet))

        #print(cleaned_tweets_training) # should return an array of cleaned sentences/Tweets

        self.vectorizer = CountVectorizer(analyzer = "word",   \
                                     tokenizer = None,    \
                                     preprocessor = None, \
                                     stop_words = None,   \
                                     max_features = 5000)

        # Output the vectorizer
        pickle.dump(self.vectorizer, open("vectorizer.sav", "wb"))

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
        train_data_features = self.vectorizer.fit_transform(cleaned_tweets_training)

        # Numpy arrays are easy to work with, so convert the result to an
        # array
        train_data_features = train_data_features.toarray()

        #print(train_data_features.shape)

        #vocab = vectorizer.get_feature_names()
        #print(vocab)

        # Sum up the counts of each vocabulary word
        #dist = np.sum(train_data_features, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the training set
        #for tag, count in zip(vocab, dist):
        #    print(count, tag)

        # Initialize a Random Forest classifier with 100 trees
        self.forest = RandomForestClassifier(n_estimators = 100, verbose=3, n_jobs=-2)

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        self.forest = self.forest.fit( train_data_features, training_data["Sentiment"] )

        # Output the forest
        pickle.dump(self.forest, open("forest.sav", "wb"))


    # Test the provided path_data against the random forest. Requires that
    # both the random forest and vectorizer is set. Outputs the results of the testing
    # Parameters: path_to_testing_data_csv - the path toe the CSV file for testing
    # Returns: none
    def test_data( self, path_to_testing_data_csv ):
        testing_data = pd.read_csv(path_to_testing_data_csv, encoding='ISO-8859-1')

        # Clean each Tweet so that each word can be analyzed
        cleaned_tweets_testing = []
        for tweet in testing_data["SentimentText"]:
            cleaned_tweets_testing.append(self.review_to_words(tweet))

        # Get a bag of words for the test set, and convert to a numpy array
        test_data_features = self.vectorizer.transform(cleaned_tweets_testing)
        test_data_features = test_data_features.toarray()

        # Use the random forest to make sentiment label predictions
        result = self.forest.predict(test_data_features)

        # Copy the results to a pandas dataframe with an "id" column and
        # a "sentiment" column
        output = pd.DataFrame( data={"id":testing_data["id"], "sentiment":result} )

        # Use pandas to write the comma-separated output file
        output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

    # Loads a saved vectorizer and a saved random forest so we can use those instead of
    # having to train the model over and over again
    # Parameters: path_to_vectorizer_sav - the .sav file for the vectorizer
    #             path_to_forest_sav - the .sav file for the random forest
    # Returns: none
    def load_vectorizer_and_random_forest( self, path_to_vectorizer_sav, path_to_forest_sav):
        self.vectorizer = pickle.load(open(path_to_vectorizer_sav, "rb"))
        self.forest = pickle.load(open(path_to_forest_sav, "rb"))
