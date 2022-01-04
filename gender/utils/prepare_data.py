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
    gender_array = []
    gender_array_clean = []

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
        gender_array.append(sent)

    
    if labels_path_untranscribed != None:
        with open(labels_path_untranscribed, 'r', encoding='utf-8') as f:
            data = f.readlines()

        for sent in data:
            sent = sent.rstrip()
            sent = sent.split('\t')[0]
            gender_array.append(sent)

    # remove the 'Other' gender selection
    for i in range(len(feature_array)):
        if gender_array[i] in ['Mies', 'Nainen']:
            feature_array_clean.append(feature_array[i])
            gender_array_clean.append(gender_array[i])


    return feature_array_clean, gender_array_clean


#def load_feat_label(feat_path, label_path, max_len, features=None):
#    feature_array = []
#    label_array = []
#    
#    if features == None:
#        features = np.load(feat_path, allow_pickle=True)
#    
#    with open(label_path, 'r', encoding='utf-8') as f:
#        label_data = f.readlines()
#
#    for i in range(len(features)):
#        if features[i].shape[0] <= max_len:    
#            feature_array.append(autograd.Variable(torch.FloatTensor(features[i])))
#            
#            label = label_data[i].rstrip()
#            label_array.append(int(label))
#            #label_array.append(label)
#    
#    label_array = torch.LongTensor(label_array)
#    return feature_array, label_array


def load_labels(features_path):
    gender_array = []
    with open(features_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    for sent in data:
        sent = sent.rstrip()
        sent = sent.split('\t')[0]
        gender_array.append([sent])

    return gender_array



def gender_to_idx(genders, gender2idx):
    res = []
    temp_res = []
    for gender in genders:
        temp_res.append(gender2idx[gender])
    
    res = torch.LongTensor(temp_res)
    return res



# extra data to be removed so that it can be divided in equal batches
def remove_extra(data, batch_size):
    extra = len(data) % batch_size
    if extra != 0:
        data = data[:-extra][:]
    return data


def combine_data(features, indexed_gender):
    res = []
    for i in range(len(features)):
        res.append((features[i], indexed_gender[i]))

    return res


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, gender = zip(*list_of_samples)

    input_seq_lengths = [len(seq) for seq in input_seqs]

    padding_value = 0

    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)
    
    gender = torch.LongTensor(gender)
    
    return pad_input_seqs, input_seq_lengths, gender
