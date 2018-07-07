import numpy as np
import re
import itertools
from collections import Counter
import csv

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
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


def load_data():
    file_reader= open('data/Emotion Phrases.csv', "rt")
    read = csv.reader(file_reader)
    data = []
    for row in read :
        x_text = [clean_str(sent) for sent in row]
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

    print(all_label)
    one_hot = np.identity(len(all_label))
    x = [one_hot[ all_label[label]-1 ] for label in x]
    x = [l.tolist() for l in x]
    x = np.array(x)
    return[y,x]



def generate_batches(data, batch_size, num_epochs, shuffle=True):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
