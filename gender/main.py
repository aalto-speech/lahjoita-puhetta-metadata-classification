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
from get_predictions import get_predictions, get_predictions_by_gender


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# load features and labels
print('Loading data...')



# Lahjoita Puhetta data
feature_paths_transcribed = ['../../data/gender/features/transcribed/train_1.npy', 
                '../../data/gender/features/transcribed/train_2.npy',
                '../../data/gender/features/transcribed/train_3.npy',
                '../../data/gender/features/transcribed/train_4.npy',
                '../../data/gender/features/transcribed/train_5.npy',
                '../../data/gender/features/transcribed/train_6.npy',
                '../../data/gender/features/transcribed/train_7.npy',
                '../../data/gender/features/transcribed/train_8.npy',
                '../../data/gender/features/transcribed/train_9.npy']


feature_paths_untranscribed = ['../../data/gender/features/untranscribed/train_1.npy', 
                '../../data/gender/features/untranscribed/train_2.npy',
                '../../data/gender/features/untranscribed/train_3.npy',
                '../../data/gender/features/untranscribed/train_4.npy',
                '../../data/gender/features/untranscribed/train_5.npy',
                '../../data/gender/features/untranscribed/train_6.npy',
                '../../data/gender/features/untranscribed/train_7.npy',
                '../../data/gender/features/untranscribed/train_8.npy',
                '../../data/gender/features/untranscribed/train_9.npy']


#features_train, gender_train = prepare_data.load_features_labels_combined(feature_paths_transcribed, feature_paths_untranscribed, '../../data/gender/labels/transcribed/train.txt', '../../data/gender/labels/untranscribed/train.txt', max_len)
#features_dev, gender_dev = prepare_data.load_features_labels_combined(['../../data/gender/features/transcribed/dev.npy'], None, '../../data/gender/labels/transcribed/dev.txt', None, max_len)
features_test, gender_test = prepare_data.load_features_labels_combined(['../../data/gender/features/transcribed/test.npy'], None, '../../data/gender/labels/transcribed/test.txt', None, max_len)


#features_train = features_test[:50]
#gender_train = gender_test[:50]
#features_dev = features_test[:50]
#gender_dev = gender_test[:50]
#features_test = features_test[:50]
#gender_test = gender_test[:50]


features_train = features_test
gender_train = gender_test
features_dev = features_test
gender_dev = gender_test
features_test = features_test
gender_test = gender_test

print('Done...')


# for gender detection
gender2idx = {'Mies': 0, 'Nainen': 1}
idx2gender = {0: 'Mies', 1: 'Nainen'}


# convert labels to indices
indexed_gender_train = prepare_data.gender_to_idx(gender_train, gender2idx)
indexed_gender_dev = prepare_data.gender_to_idx(gender_dev, gender2idx)
indexed_gender_test = prepare_data.gender_to_idx(gender_test, gender2idx)


# combine features and gender in a tuple
train_data = prepare_data.combine_data(features_train, indexed_gender_train)
dev_data = prepare_data.combine_data(features_dev, indexed_gender_dev)
test_data = prepare_data.combine_data(features_test, indexed_gender_test)


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



x_vector_model = XVectorModel(features_train[0].size(1), 512, len(gender2idx)).to(device)

# create weights for the classes
#class_sample_count = np.unique(gender_train, return_counts=True)[1]
#weight = 1. / class_sample_count
#weight = torch.from_numpy(weight).float().to(device)


# train
if skip_training == False:
    print('Training...')
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(x_vector_model.parameters(), lr=lr)

    #checkpoint = torch.load('weights/gender/state_dict_17.pt', map_location=torch.device('cpu'))
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


batch_size = 1

pairs_batch_test = DataLoader(dataset=test_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


get_predictions_by_gender(x_vector_model, batch_size, idx2gender, pairs_batch_test)
