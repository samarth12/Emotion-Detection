import numpy as np
import re
import itertools
from collections import Counter
import cPickle as pickle
import os
import csv
#input = ["fear,Every time I imagine that someone I love or I could contact a serious illness, even death", "angry, I am an angry man"]

def clean_string(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

file_reader= open('data/Emotion Phrases.csv', "rt")
read = csv.reader(file_reader)
data = []
for row in read :
    x_text = [clean_string(sent) for sent in row]
    data.append(x_text)


#print(data)
#x_text = [clean_string(sent) for sent in data]
x = [x_text[0] for x_text in data]
y = [x_text[1] for x_text in data]
    #print(y)
#print(x)
#print(x)
all_label = dict()
for label in x:
    if not label in all_label:
        all_label[label] = len(all_label) + 1

#print(all_label)
one_hot = np.identity(len(all_label))
x = [one_hot[ all_label[label]-1 ] for label in x]
x = [l.tolist() for l in x]
x = np.array(x)
print(x)
#print(x[0])
#print(x)


"""
x_text = list(open('data/Emotion Phrases.csv').readlines())
print(x_text)
x_text = [clean_string(sent) for sent in x_text]
y = [s.split(' ')[0].split(',')[0] for s in x_text]
#print(y)
#print(x_text)
#print(read)
"""
#data =
"""
out = clean_string(input)
y = [s.split(' ')[0].split(',')[0] for s in out]
print(y)
x = out.split(" ")[2:]

all_label = dict()
for label in y:
    if not label in all_label:
        all_label[label] = len(all_label) + 1
print(x)
"""
