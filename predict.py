import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
import os
import csv
import itertools
import operator

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
import keras.models
import pickle as pkl
import lstm

from keras.models import load_model
path = "text_lstm_weights.h5"


np.random.seed(7)

DIR_GLOVE = 'glove/'
#DIR_DATA = 'data/'
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1

def createEmbeddingMatrix(word_index,embeddings_index):
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("In the emb func")
    print(np.shape(embedding_matrix))
    mat = x = np.delete(embedding_matrix, slice(8999,9016), axis=0)
    print("In the emb func NEWWW")
    print(np.shape(mat))
    return mat

sentences,labels, labs = lstm.loadData('data/Emotion Phrases.csv')
print("Labels")
print(labs)
test_d1 =[]
test_d1.append("After my girlfriend had taken her exam we went to her parent's place.")
test_d1.append("When I had my children.")
print("CHECK HERE")
print(test_d1)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(test_d1)
test_d = tokenizer.texts_to_sequences(test_d1)
print("Testing")
print(test_d)
test_d = pad_sequences(test_d, maxlen=MAX_SEQUENCE_LENGTH)
vector = np.array([test_d.flatten()])
print("VECTOR")
print(vector)
embeddings = lstm.gloveVec('glove.6B.300d.txt')
print("HEREEEE")
print(np.shape(embeddings))
vocab, data = lstm.createVocabAndData(sentences)
embedding_mat = createEmbeddingMatrix(vocab,embeddings)
print(np.shape(embedding_mat))
pickle.dump([data, labels, embedding_mat], open('embedding_matrix.pkl', 'wb'))
print ("Data created")

print("Train Test split")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SPLIT, random_state=42)


def create_model(embedding_matrix):
    model = Sequential()
    n, embedding_dims = embedding_matrix.shape

    model.add(Embedding(n, embedding_dims, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    model.add(LSTM(128, dropout=0.6, recurrent_dropout=0.6))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    return model

def load_trained_model(weights_path):
    model = create_model(embedding_mat)
    model.load_weights(weights_path)
    yay = model.predict(test_d)
    print("Here is your probability")
    print(yay)

    t = 0

    for text in test_d1:
        i = 0
        print("Prediction for \"%s\": " % (text))
        for label in labs:
            print("\t%s ==> %f" % (label, yay[t][i]))
            i = i + 1
        t = t + 1
    for i in range(len(yay)):
    #final = yay[0]
        q =0
        for labe in labs:
            labs[labe] = yay[i][q]
            q = q+1
        print(labs)
        newA = dict(sorted(labs.iteritems(), key=operator.itemgetter(1), reverse=True)[:2])
        print("JUST SEE HERE")
        print(newA)

    #classes = np.argmax(yay)
    #print(classes)

load_trained_model(path)
