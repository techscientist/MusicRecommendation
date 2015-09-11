#! /usr/bin/env python
"""
@author: ruonan
This file load and parse raw csv data and dump them as binary formate so we can reload the parsed tables during the training. 
"""

import csv
import cPickle as pickle

import numpy as np

import users
from numpy import dtype
from numpy.oldnumeric.random_array import permutation
from _random import Random
from math import ceil
users.load_users('data/users.pkl')

import words
words.load_words('data/words.pkl')

from collections import defaultdict

def indicator(cat_id, dim):
    v = [0] * dim
    if cat_id >= 0:
        v[cat_id] = 1.0
    return v

def binary(w):
    if w == -1:
        return 0
    if w == 0:
        return -1.0
    if w == 1:
        return 1.0

def label(example):
    return float(example['rating'])


def read_examples(csv_filename):
    csv_file = open(csv_filename, 'rb')
    reader = csv.reader(csv_file)
    reader.next()  # ignore header
    examples = []
    for row in reader:
        example = {}
        example['artist']   = int(row[0])
        example['track']    = int(row[1])
        example['user']     = int(row[2])
        if len(row) == 5:  # train
            example['rating'] = int(row[3])
        else:  # test
            example['rating'] = -1
        example['time']     = int(row[-1])
        examples.append(example)
    csv_file.close()
    return examples

def write_examples(csv_filename, examples):
    csv_file = open(csv_filename, 'wb')
    writer = csv.writer(csv_file)
    writer.writerow(['Artist','Track','User','Rating','Time'])  # header
    for example in examples:
        row = [str(example['artist']),
               str(example['track']),
               str(example['user']),
               str(example['rating']),
               str(example['time'])]
        writer.writerow(row)
    csv_file.close()

def load_examples(pkl_filename):
    pkl_file = open(pkl_filename, 'rb')
    examples = pickle.load(pkl_file)
    pkl_file.close()
    return examples

def save_examples(pkl_filename, examples):
    pkl_file = open(pkl_filename, 'wb')
    pickle.dump(examples, pkl_file, -1)
    pkl_file.close()

def represent(example):
    vector = []
    artist_id = example['artist']
    track_id  = example['track']
    user_id   = example['user']
    time_id   = example['time']
    vector.extend(indicator(artist_id, 50))
    vector.extend(indicator(track_id, 184))
    vector.extend(indicator(time_id, 24))
    user = users.user_dict.get(user_id, defaultdict(lambda: -1.0))
    vector.append(binary(user['gender']))
    vector.append(user['age']/100.0)
    vector.extend(indicator(user['working'], 13))
    vector.extend(indicator(user['region'], 4))
    vector.extend(indicator(user['music'], 5))
    vector.append(user['list_own'] /24.0)
    vector.append(user['list_back']/24.0)
    vector.extend([user['q%d' % (j+1)]/100.0 for j in range(19)])
    word = words.word_dict.get((artist_id, user_id), defaultdict(lambda: -1))
    vector.extend(indicator(word['heard-of'], 4))
    vector.extend(indicator(word['own-artist-music'], 5))
    vector.append(word['like-artist']/100.0)
    vector.extend([binary(word['w%d' % (j+1)]) for j in range(81)])
    vector.append(user_id/50928.0)
    # The principled way is to use vector.extend(indicator(user_id, 50928)), 
    # but it would take too much memory.
    
    index = 1
    pairs = []
    for v in vector:
        if v == 0:
            index +=1
            continue
        else:
            pairs.append("%d:%f"%(index,v))
            index +=1
            
    return pairs
    
def shuffle(data):
    import random
    l = len(data)
    for i in xrange(l):
        n = l-i - 1
        m = int(ceil(random.random() * n))
        tmp = data[m]
        data[m] = data[n]
        data[n] = tmp
        
def save_libsvm(data_filename, examples):
    
    shuffle(examples)
        
    fileTest = open("%s.%s"%(data_filename,"test"),'w')
    fileTrain = open("%s.%s"%(data_filename,"train"),'w')

    count = 0
    
    # write to libsvm file
    for e in examples:                   
        count += 1
        label =  str(e['rating'])
        features = represent(e)
        
        sep_line = [label]
        sep_line.extend(features)
        sep_line.append('\n')
        
        line = ' '.join(sep_line)
        
        if count % 20 ==0:
            fileTest.write(line)
        else:
            fileTrain.write(line)
        
    fileTest.close()
    fileTrain.close()

if __name__ == "__main__":
    train_examples = read_examples('data/train.csv')
    save_examples('data/train.pkl', train_examples)
    test_examples = read_examples('data/test.csv')
    save_examples('data/test.pkl', test_examples)
