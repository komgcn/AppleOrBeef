import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random

#fixed lexicon with 5 fruits and 5 meats
lemmatizer = WordNetLemmatizer()
lexicon = ['fish','apple', 'orange', 'beef','lamb','banana','chicken','pear','cherry','duck']
test_size = 0.1

#convert a .txt file into list of lexicon and classification
def handle_sample(sample,classification):

    feature_set = []
    counter = 0
    with open(sample, 'r') as f:
        content = f.readlines()
        for l in content:
            words = word_tokenize(l.lower())
            words = [lemmatizer.lemmatize(i) for i in words]
            features = np.zeros(len(lexicon))
            for w in words:
                if w.lower() in lexicon:
                    index = lexicon.index(w.lower())
                    features[index] += 1
            features = list(features)
            feature_set.append([features,classification])
            counter += 1
    print(counter,classification,' feature_set returned')
    return feature_set

#function called by other scripts to return the whole sets of data for neural network training and testing
def create_sets(apple,beef):

    features = []
    features += handle_sample(apple,[1,0])
    features += handle_sample(beef,[0,1])
    random.shuffle(features)

    features = np.array(features)
    testing_size = int(len(features) * test_size)

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    print('train_x,y,test_x,y returned')
    return train_x, train_y, test_x, test_y

#create_sets('apple.txt','beef.txt')