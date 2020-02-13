import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)

DATA_JSON_FILE = '/home/shyam/GitRespository/MachineLearningUdumy/Pre-ProcessTextDataNaiveBayesClassifier/SampleData' \
                 '/SpamDatanext/SpamData/01_Processing/email-text-data.json'
data = pd.read_json(DATA_JSON_FILE)
vector = CountVectorizer(stop_words='english')
all_feature = vector.fit_transform(data.MESSAGE)
print(f'shape of all features {all_feature.shape}')
print(f'vocabulary data {vector.vocabulary_}')
# train and slit data
x_train, x_test, y_train, y_test = train_test_split(all_feature, data.CATEGORY, test_size=0.3, random_state=88)
classifier = MultinomialNB()
classifier.fit(x_train, y_train)
nr_correct = (y_test == classifier.predict(x_test)).sum()
nr_incorrect = y_test.size - nr_correct
fraction_wrong = nr_incorrect / (nr_correct + nr_incorrect)
print(f'number of incorrect classified {nr_incorrect}')
print(f'number of correct document classified correct {nr_correct}')
print(f'the testing accuracy of model is  {fraction_wrong}')
