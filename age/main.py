import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np
import pickle
import math

import utils.prepare_data as prepare_data
from model import XVectorModel, Transformer
from config.config import *
from train import train
from get_predictions import get_predictions, get_predictions_relaxed, get_predictions_by_age


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# load features and labels
print('Loading data...')


# Lahjoita Puhetta data
feature_paths_transcribed = ['../../data/age/features/transcribed/train_1.npy', 
                '../../data/age/features/transcribed/train_2.npy',
                '../../data/age/features/transcribed/train_3.npy',
                '../../data/age/features/transcribed/train_4.npy',
                '../../data/age/features/transcribed/train_5.npy',
                '../../data/age/features/transcribed/train_6.npy',
                '../../data/age/features/transcribed/train_7.npy',
                '../../data/age/features/transcribed/train_8.npy']


feature_paths_untranscribed = ['../../data/age/features/untranscribed/train_1.npy', 
                '../../data/age/features/untranscribed/train_2.npy',
                '../../data/age/features/untranscribed/train_3.npy',
                '../../data/age/features/untranscribed/train_4.npy',
                '../../data/age/features/untranscribed/train_5.npy',
                '../../data/age/features/untranscribed/train_6.npy',
                '../../data/age/features/untranscribed/train_7.npy',
                '../../data/age/features/untranscribed/train_8.npy',
                '../../data/age/features/untranscribed/train_9.npy']


#features_train, age_train = prepare_data.load_features_labels_combined(feature_paths_transcribed, feature_paths_untranscribed, '../../data/age/labels/transcribed/train.txt', '../../data/age/labels/untranscribed/train.txt', max_len)
#features_dev, age_dev = prepare_data.load_features_labels_combined(['../../data/age/features/transcribed/dev.npy'], None, '../../data/age/labels/transcribed/dev.txt', None, max_len)
features_test, age_test = prepare_data.load_features_labels_combined(['../../data/age/features/transcribed/test.npy'], None, '../../data/age/labels/transcribed/test.txt', None, max_len)



#features_train = features_test[:500]
#age_train = age_test[:500]
#features_dev = features_test[:500]
#age_dev = age_test[:500]
#features_test = features_test[:500]
#age_test = age_test[:500]


features_train = features_test
age_train = age_test
features_dev = features_test
age_dev = age_test
features_test = features_test
age_test = age_test

print('Done...')


age2idx = {'1-10': 0, '11-20': 1, '21-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7, '81-90': 8, '91-100': 9, '101+': 10}
idx2age = {v: k for k, v in age2idx.items()}


# convert labels to indices
indexed_age_train = prepare_data.age_to_idx(age_train, age2idx)
indexed_age_dev = prepare_data.age_to_idx(age_dev, age2idx)
indexed_age_test = prepare_data.age_to_idx(age_test, age2idx)


# combine features and age in a tuple
train_data = prepare_data.combine_data(features_train, indexed_age_train)
dev_data = prepare_data.combine_data(features_dev, indexed_age_dev)
test_data = prepare_data.combine_data(features_test, indexed_age_test)



pairs_batch_train = DataLoader(dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)

pairs_batch_dev = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    drop_last=True,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)

pairs_batch_dev_acc = DataLoader(dataset=dev_data,
                    batch_size=1,
                    drop_last=False,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


x_vector_model = XVectorModel(features_train[0].size(1), 512, len(age2idx)).to(device)

# train
if skip_training == False:
    print('Training...')
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(x_vector_model.parameters(), lr=lr)

    #checkpoint = torch.load('weights/x_vector/age_1/state_dict_28.pt', map_location=torch.device('cpu'))
    #x_vector_model.load_state_dict(checkpoint['x_vector_model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

    train(pairs_batch_train, 
            pairs_batch_dev,
            pairs_batch_dev_acc,
            x_vector_model,
            criterion,
            optimizer,
            num_epochs,
            batch_size,
            len(features_train),
            len(features_dev),
            device) 


pairs_batch_test = DataLoader(dataset=test_data,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


get_predictions(x_vector_model, batch_size, idx2age, pairs_batch_test)
