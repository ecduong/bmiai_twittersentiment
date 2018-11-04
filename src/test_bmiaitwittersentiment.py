from BMIAITwitterSentiment import BMIAITwitterSentiment as bmiai

PATH_TO_TRAINING_DATA = "../data/train.csv"
PATH_TO_TESTING_DATA = "../data/test.csv"
PATH_TO_VECTORIZER = "vectorizer.sav"
PATH_TO_FOREST = "forest.sav"

bmiai.train_data(bmiai, PATH_TO_TRAINING_DATA)
bmiai.load_vectorizer_and_random_forest(bmiai, PATH_TO_VECTORIZER, PATH_TO_FOREST)
bmiai.test_data(bmiai, PATH_TO_TESTING_DATA)
