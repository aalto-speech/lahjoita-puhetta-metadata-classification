import numpy as np
import os

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence
from random import randrange
import math


# load features combined in one file
def load_features_labels_transcripts_combined(feature_paths_transcribed, feature_paths_untranscribed, labels_path, labels_path_untranscribed, transcripts_path, max_len):
    feature_array = []
    dialect_array = []
    transcript_array = []
    feature_clean_array = []
    dialect_clean_array = []
    transcript_clean_array = []

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

    
    # load transcripts
    if transcripts_path != None:
        with open(transcripts_path, 'r', encoding='utf-8') as f:
            data = f.readlines()

        for sent in data:
            sent = sent.rstrip()
            transcript_array.append(sent)


    # load labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    for sent in data:
        sent = sent.rstrip()
        sent = sent.split('\t')[0]
        dialect_array.append(int(sent))

    
    if labels_path_untranscribed != None:
        with open(labels_path_untranscribed, 'r', encoding='utf-8') as f:
            data = f.readlines()

        for sent in data:
            sent = sent.rstrip()
            sent = sent.split('\t')[0]
            dialect_array.append(int(sent))

    
    # use only samples less than 50s
    #for i in range(len(feature_array)):
    #    if feature_array[i].size(0) < max_len:
    #        feature_clean_array.append(feature_array[i])
    #        dialect_clean_array.append(dialect_array[i])
    #        transcript_clean_array.append(transcript_array[i])

    #return feature_clean_array, transcript_clean_array, dialect_clean_array
    return feature_array, transcript_array, dialect_array



# load features combined in one file
def load_features_combined(features_path, max_len):
    feature_array = []
    features = np.load(features_path, allow_pickle=True)
    
    for feature in features:
        feature = feature[:max_len]
        feature_array.append(autograd.Variable(torch.FloatTensor(feature)))
        #feature_array.append(feature)

    return feature_array


def load_labels(labels_path):
    dialect_array = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    for sent in data:
        sent = sent.rstrip()
        sent = sent.split('\t')
        
        dialect_array.append(torch.LongTensor([int(sent[0])]))

    return dialect_array


def load_feat_label(feat_path, label_path, max_len, features=None):
    feature_array = []
    label_array = []
    
    if features == None:
        features = np.load(feat_path, allow_pickle=True)
    
    with open(label_path, 'r', encoding='utf-8') as f:
        label_data = f.readlines()

    for i in range(len(features)):
        if features[i].shape[0] <= max_len:    
            feature_array.append(autograd.Variable(torch.FloatTensor(features[i])))
            
            label = label_data[i].rstrip()
            label_array.append(torch.LongTensor([int(label)]))
    
    return feature_array, label_array


def load_feat_trn_label(feat_path, trn_path, label_path, max_len, features=None):
    feature_array = []
    trn_array = []
    label_array = []
    
    if features == None:
        features = np.load(feat_path, allow_pickle=True)
    
    with open(trn_path, 'r', encoding='utf-8') as f:
        trn_data = f.readlines()
    
    with open(label_path, 'r', encoding='utf-8') as f:
        label_data = f.readlines()

    for i in range(len(features)):
        if features[i].shape[0] <= max_len:    
            feature_array.append(autograd.Variable(torch.FloatTensor(features[i])))
            
            trn = trn_data[i].rstrip()
            trn_array.append([trn])

            label = label_data[i].rstrip()
            label_array.append(torch.LongTensor([int(label)]))
    
    return feature_array, trn_array, label_array



# for labels with text
def load_labels_trn(trn_path, dialect_path):
    dialect_array = []
    trn_array = []
    with open(trn_path, 'r', encoding='utf-8') as f:
        trn_data = f.readlines()
    
    with open(dialect_path, 'r', encoding='utf-8') as f:
        dialect_data = f.readlines()

    for i, sent in enumerate(trn_data):
        sent = trn_data[i].rstrip()
        dialect = dialect_data[i]
        trn_array.append([sent])
        dialect_array.append(torch.LongTensor([int(dialect)]))
    
    return trn_array, dialect_array


# loads features and dialects and splits them in 15 seconds
def load_feat_labels(features_path, labels_path, max_len, pre_loaded):
    feature_array = []
    label_array = []

    if pre_loaded == True:
        features = features_path
    else:
        features = np.load(features_path, allow_pickle=True)

 
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    
    for i in range(len(features)):
        num_chunks = math.ceil(features[i].shape[0] / max_len)
        for chunk in range(1, num_chunks+1):
            feature = features[i][chunk-1*max_len : max_len*chunk]
            
            if chunk == 0 and num_chunks == 1:
                feature_array.append(autograd.Variable(torch.FloatTensor(features[i])))
                label = labels[i].rstrip()
                label_array.append(torch.LongTensor([int(label)]))
            
            if feature.shape[0] >= 1000 and num_chunks > 1:
                feature_array.append(autograd.Variable(torch.FloatTensor(feature)))
                label = labels[i].rstrip()
                label_array.append(torch.LongTensor([int(label)]))
            
    return feature_array, label_array



def extract_bert_embeddings(model, tokenizer, data, device):
    sentence = []
    words = []
    for sent in data:
        sent = '[CLS] ' + sent + ' [SEP]'

        # tokenize it
        tokenized_sent = tokenizer.tokenize(sent)
        if len(tokenized_sent) > 512:
            tokenized_sent = tokenized_sent[:511]
            tokenized_sent.append('[SEP]')
        sent_idx = tokenizer.convert_tokens_to_ids(tokenized_sent)

        # add segment ID
        segments_ids = [1] * len(sent_idx)

        # convert data to tensors
        sent_idx = torch.tensor([sent_idx]).to(device)
        segments_ids = torch.tensor([segments_ids]).to(device)
        
        #get embeddings
        with torch.no_grad():
            outputs = model(sent_idx, segments_ids)
            hidden_states = outputs[2]
            hidden_states = torch.stack(hidden_states, dim=0)
            embeddings = torch.sum(hidden_states[-4:], dim=0)
            embeddings = embeddings.squeeze()
            sentence.append(embeddings.to('cpu'))

    return sentence


def combine_data(features, indexed_label, use_trn, indexed_trn=None):
    res = []
    if use_trn == True:
        for i in range(len(features)):
            res.append((features[i], indexed_trn[i], indexed_label[i]))
    else:
        for i in range(len(features)):
            res.append((features[i], indexed_label[i]))

    return res


# used with transcripts
def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, trn_seqs, dialect = zip(*list_of_samples)

    input_seq_lengths = [len(seq) for seq in input_seqs]

    padding_value = 0

    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)
    
    # pad transcripts
    pad_trn_seqs = pad_sequence(trn_seqs, padding_value=padding_value)
    
    dialect = torch.LongTensor(dialect)
    
    return pad_input_seqs, input_seq_lengths, pad_trn_seqs, dialect


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    input_seqs, dialect = zip(*list_of_samples)

    input_seq_lengths = [len(seq) for seq in input_seqs]

    padding_value = 0

    # pad input sequences
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)
    
    dialect = torch.LongTensor(dialect)
    
    return pad_input_seqs, input_seq_lengths, dialect
