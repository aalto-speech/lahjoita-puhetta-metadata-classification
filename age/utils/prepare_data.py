import numpy as np
import os

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence
from random import randrange


# load features combined in one file
def load_features_labels_combined(feature_paths_transcribed, feature_paths_untranscribed, labels_path, labels_path_untranscribed, max_len):
    feature_array = []
    feature_array_clean = []
    age_array = []
    age_array_clean = []

    # load features
    for features_path in feature_paths_transcribed:
        features = np.load(features_path, allow_pickle=True, encoding='bytes')
    
        for feature in features:
            feature = feature[:max_len]
            feature_array.append(autograd.Variable(torch.FloatTensor(feature)))
        
    if feature_paths_untranscribed != None:
        for features_path in feature_paths_untranscribed:
            features = np.load(features_path, allow_pickle=True, encoding='bytes')
    
            for feature in features:
                feature = feature[:max_len]
                feature_array.append(autograd.Variable(torch.FloatTensor(feature)))

    
    # load labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    for sent in data:
        sent = sent.rstrip()
        sent = sent.split('\t')[0]
        age_array.append(sent)

    
    if labels_path_untranscribed != None:
        with open(labels_path_untranscribed, 'r', encoding='utf-8') as f:
            data = f.readlines()

        for sent in data:
            sent = sent.rstrip()
            sent = sent.split('\t')[0]
            age_array.append(sent)   

    return feature_array, age_array


# load features combined in one file
def load_features_combined(features_path, max_len):
    feature_array = []
    features = np.load(features_path, allow_pickle=True)
    
    for feature in features:
        feature = feature[:max_len]
        feature_array.append(autograd.Variable(torch.FloatTensor(feature)))

    return feature_array


def load_labels(features_path):
    age_array = []
    with open(features_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    for sent in data:
        sent = sent.rstrip()
        sent = sent.split('\t')[1]
        age_array.append([sent])

    return age_array



def age_to_idx(ages, age2idx):
    res = []
    temp_res = []
    for age in ages:
        temp_res.append(age2idx[age])
    
    res = torch.LongTensor(temp_res)
    return res



# extra data to be removed so that it can be divided in equal batches
def remove_extra(data, batch_size):
    extra = len(data) % batch_size
    if extra != 0:
        data = data[:-extra][:]
    return data


def combine_data(features, indexed_age):
    res = []
    for i in range(len(features)):
        res.append((features[i], indexed_age[i]))

    return res


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, age = zip(*list_of_samples)

    input_seq_lengths = [len(seq) for seq in input_seqs]

    padding_value = 0

    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)
    
    age = torch.LongTensor(age)
    
    return pad_input_seqs, input_seq_lengths, age
