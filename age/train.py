import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.calculate_f1 import print_scores
from utils.beam_search import Generator
from sklearn.metrics import accuracy_score

import pickle
import gc

def train(pairs_batch_train, pairs_batch_dev, pairs_batch_dev_acc, x_vector_model, criterion, optimizer, num_epochs, batch_size, train_data_len, dev_data_len, device):
    accumulate_steps = 1
    
    for epoch in range(1, 41):
        all_predictions = []
        all_labels = []
        all_tag_predictions = []
        all_tags = []

        x_vector_model.eval()

        batch_loss_train = 0 
        batch_loss_dev = 0
    
        
        for iteration, batch in enumerate(pairs_batch_train):
            optimizer.zero_grad()
            train_loss = 0
            
            pad_input_seqs, input_seq_lengths, age_seqs = batch
            pad_input_seqs, age_seqs = pad_input_seqs.to(device), age_seqs.to(device)
            
            output = x_vector_model(pad_input_seqs)

            train_loss = criterion(output, age_seqs)
            train_loss.backward()
            batch_loss_train += train_loss.detach().item() * output.size()[0]
            optimizer.step()
            
            
        # VALIDATION
        with torch.no_grad():
            x_vector_model.eval()

            for iteration, batch in enumerate(pairs_batch_dev):
                dev_loss = 0
                
                pad_input_seqs, input_seq_lengths, age_seqs = batch
                pad_input_seqs, age_seqs = pad_input_seqs.to(device), age_seqs.to(device)
            
                output = x_vector_model(pad_input_seqs)

                dev_loss = criterion(output, age_seqs)
                batch_loss_dev += dev_loss.item() * output.size()[0]
               
        # calculate accuracy
        true_age = []
        predicted_age = []

        for l, batch in enumerate(pairs_batch_dev_acc):
            pad_input_seqs, input_seq_lengths, age_seqs = batch
            pad_input_seqs, age_seqs = pad_input_seqs.to(device), age_seqs.to(device)
            output = x_vector_model(pad_input_seqs)
            output = F.softmax(output, dim=-1)
            topi, topk = output.topk(1)
            
            true_age.append(age_seqs.item())
            predicted_age.append(topk.item())
        
        
        accuracy = accuracy_score(true_age, predicted_age)


        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f   acc: %.4f' % (epoch, (batch_loss_train / len(pairs_batch_train)), (batch_loss_dev / len(pairs_batch_dev)), accuracy))

        with open('loss/model_50_seconds.txt', 'a') as f:
            f.write(str(epoch) + '  ' + str(batch_loss_train / len(pairs_batch_train)) + '  ' + str(batch_loss_dev / len(pairs_batch_dev)) + '  ' + str(accuracy) + '\n')


        print('saving the model...')
        torch.save({
        'x_vector_model': x_vector_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, 'weights/model_50_seconds/state_dict_' + str(epoch) + '.pt')
