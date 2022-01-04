import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertModel

import numpy as np
import pickle
import math
import fasttext

import utils.prepare_data as prepare_data
from model import XVectorModel
from config.config import *
from train import train
from get_predictions import get_predictions, get_predictions_by_topic


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# load features and labels
print('Loading data...')


# Lahjoita Puhetta data
feature_paths_transcribed = ['../../data/topic/features/asr_transcribed/train_1.npy', 
                '../../data/topic/features/asr_transcribed/train_2.npy',
                '../../data/topic/features/asr_transcribed/train_3.npy',
                '../../data/topic/features/asr_transcribed/train_4.npy',
                '../../data/topic/features/asr_transcribed/train_5.npy',
                '../../data/topic/features/asr_transcribed/train_6.npy',
                '../../data/topic/features/asr_transcribed/train_7.npy',
                '../../data/topic/features/asr_transcribed/train_8.npy',
                '../../data/topic/features/asr_transcribed/train_9.npy']
                #'../../data/topic/features/transcribed/train_10.npy']


feature_paths_untranscribed = ['../../data/topic/features/untranscribed/train_1.npy', 
                '../../data/topic/features/untranscribed/train_2.npy',
                '../../data/topic/features/untranscribed/train_3.npy',
                '../../data/topic/features/untranscribed/train_4.npy',
                '../../data/topic/features/untranscribed/train_5.npy',
                '../../data/topic/features/untranscribed/train_6.npy',
                '../../data/topic/features/untranscribed/train_7.npy',
                '../../data/topic/features/untranscribed/train_8.npy',
                '../../data/topic/features/untranscribed/train_9.npy',
                '../../data/topic/features/untranscribed/train_10.npy']


#features_train, trn_train, labels_train = prepare_data.load_features_labels_transcripts_combined(feature_paths_transcribed, None, '../../data/topic/labels/asr_transcribed/train.txt', None, '../../data/topic/transcripts/asr_transcribed/train.txt', max_len)
#features_dev, trn_dev,  labels_dev = prepare_data.load_features_labels_transcripts_combined(['../../data/topic/features/transcribed/dev.npy'], None, '../../data/topic/labels/transcribed/dev.txt', None, '../../data/topic/transcripts/dev.txt', max_len)
features_test, trn_test, labels_test = prepare_data.load_features_labels_transcripts_combined(['../../data/topic/features/transcribed/test.npy'], None, '../../data/topic/labels/transcribed/test.txt', None, '../../data/topic/transcripts/test.txt', max_len)


#features_train = features_test[:50]
#labels_train = labels_test[:50]
#trn_train = trn_test[:50]
#features_dev = features_test[:50]
#labels_dev = labels_test[:50]
#trn_dev = trn_test[:50]
#features_test = features_test[:50]
#labels_test = labels_test[:50]
#trn_test = trn_test[:50]

features_train = features_test
labels_train = labels_test
trn_train = trn_test
features_dev = features_test
labels_dev = labels_test
trn_dev = trn_test

print('Done...')



# convert words to indices
if use_trn == True:
    # extract BERT embeddings
    model = BertModel.from_pretrained("TurkuNLP/bert-base-finnish-uncased-v1", output_hidden_states=True).to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-uncased-v1")
    
    trn_train = prepare_data.extract_bert_embeddings(model, tokenizer, trn_train, device)
    trn_dev = prepare_data.extract_bert_embeddings(model, tokenizer, trn_dev, device)
    trn_test = prepare_data.extract_bert_embeddings(model, tokenizer, trn_test, device)
    # DONE



topic2idx = {'Eläimet': 0, 'K-18': 1, 'Kerro': 2, 'Kesä': 3, 'Luonto': 4, 'Lähelläni': 5, 'Mediataidot': 6, 'Urheilu': 7}
idx2topic = {v: k for k, v in topic2idx.items()}


# convert labels to indices
labels_train = prepare_data.topic_to_idx(labels_train, topic2idx)
labels_dev = prepare_data.topic_to_idx(labels_dev, topic2idx)
labels_test = prepare_data.topic_to_idx(labels_test, topic2idx)



# combine features and labels in a tuple
if use_trn == True:
    train_data = prepare_data.combine_data(features_train, labels_train, use_trn, indexed_trn=trn_train)
    dev_data = prepare_data.combine_data(features_dev, labels_dev, use_trn, indexed_trn=trn_dev)
    test_data = prepare_data.combine_data(features_test, labels_test, use_trn, indexed_trn=trn_test)
else:
    train_data = prepare_data.combine_data(features_train, labels_train, use_trn)
    dev_data = prepare_data.combine_data(features_dev, labels_dev, use_trn)
    test_data = prepare_data.combine_data(features_test, labels_test, use_trn)


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
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)



x_vector_model = XVectorModel(features_train[0].size(1), 512, len(topic2idx), use_trn).to(device)


# train
if skip_training == False:
    print('Training...')
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(x_vector_model.parameters(), lr=lr)
    
    #checkpoint = torch.load('weights/model_trn/state_dict_21.pt', map_location=torch.device('cpu'))
    #x_vector_model.load_state_dict(checkpoint['x_vector_model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(x_vector_model)

    train(pairs_batch_train, 
            pairs_batch_dev,
            pairs_batch_dev_acc,
            x_vector_model,
            criterion,
            optimizer,
            batch_size,
            device,
            use_trn)


batch_size = 1

pairs_batch_test = DataLoader(dataset=test_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


get_predictions(x_vector_model, batch_size, pairs_batch_test, use_trn)
