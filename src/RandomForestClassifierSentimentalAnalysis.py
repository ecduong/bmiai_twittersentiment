import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
#import tflearn
#from tflearn.data_utils import to_categorical, pad_sequences

# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
PATH_TO_TRAINING_DATA = "../data/train.csv"
PATH_TO_TESTING_DATA = "../data/test.csv"
#Dataset loading
train = pd.read_csv(PATH_TO_TRAINING_DATA, encoding = "ISO-8859-1")
test = pd.read_csv(PATH_TO_TESTING_DATA, encoding = "ISO-8859-1")

##########################################################################################
#simple cleaning text example!
#train1 = BeautifulSoup(train["SentimentText"][171], "lxml")
#print(train1.get_text())
#letters_only = re.sub("[^a-zA-Z]"," ", train1.get_text()) #decide on whether to change it to include alphanumerical

#lower_case = letters_only.lower()
#words = lower_case.split()

#print(letters_only)

#words = [w for w in words if not w in stopwords.words("english")]
#print(words)

def tweets_to_words(raw_tweets):
    #1. Remove HTML
    tweet_text = BeautifulSoup(raw_tweets,"lxml").get_text()
    
    #2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", tweet_text)
    
    #3. conver to lower case, split into individual words
    words = letters_only.lower().split()
    
    #4.convert stop words to a set since it is faster to search a set than a list
    stops = set(stopwords.words("english"))
    
    #5.remove stop words
    meaningful_words = [w for w in words if not w in stops]
    
    #6.Join the words back into one string separated by space,
    #and return the result
    return(" ".join(meaningful_words))

clean_tweets = tweets_to_words(train["SentimentText"][171])

#Get number of tweets
num_reviews = train["SentimentText"].size
print(num_reviews)

#Initialize an empty list
clean_train_tweets = []

print("Cleaning and parsing the training set movie reviews... \n")
for i in range(0, num_reviews):
    #if the index is evenly divisible by 1000, print a message
    if ( (i+1)%1000 == 0):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_tweets.append( tweets_to_words(train["SentimentText"][i]))
    
######################################################################################
    
print("Creating the bag of words... \n")


#Initialize the "CountVectorizer" object, which is scikit-learn's
#bag of words tool.
vectorizer = CountVectorizer(analyzer = "word", \
                             tokenizer = None, \
                             preprocessor = None, \
                             stop_words = None, \
                             max_features = 5000 #might need to decrease this number later
                            )

#Initialize a Random Forest Classifier with 100 trees will probably lagg your computer LOL runs very slow took 18mins
#forest = RandomForestClassifier(n_estimators = 100,verbose=3,n_jobs=-2)

#fit_transform(): fits model and learns vocabulary and transform our training data into feature vectors.
#input to fit_transform should be a list of strings.
train_data_features = vectorizer.fit_transform(clean_train_tweets)

#numpy arrays are easy to work with, so conver the result to an array
train_data_features = train_data_features.toarray()

########################################################################################

print("Training the random forest...")
#from sklearn.ensemble import RandomForestClassifier

#Initialize a Random Forest Classifier with 100 trees will probably lagg your computer LOL runs very slow took 18mins
forest = RandomForestClassifier(n_estimators = 100,verbose=3,n_jobs=-2)


#fit the forest to the training set, using the bag of words as 
#features and the sentiment labels as the response variable
#
#this may take a few minutes to run
forest = forest.fit( train_data_features, train["Sentiment"])

############################################################################
print(test.shape)

num_reviews = test["SentimentText"].size

clean_tweet_reviews = []
print("Cleaning and parsing the test set.. \n") 
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_tweet = tweets_to_words( test["SentimentText"][i] )
    clean_tweet_reviews.append( clean_tweet )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_tweet_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"ItemID":test["ItemID"], "Sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "submission.csv", index=False, quoting=3 )



