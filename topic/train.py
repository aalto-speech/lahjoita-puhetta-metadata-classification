import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.calculate_f1 import print_scores
from utils.beam_search import Generator
from sklearn.metrics import accuracy_score

import pickle
import gc

def train(pairs_batch_train, pairs_batch_dev, pairs_batch_dev_acc, x_vector_model,  criterion, optimizer, batch_size, device, use_trn):
    for epoch in range(1, 41):
        all_predictions = []
        all_labels = []
        all_tag_predictions = []
        all_tags = []

        x_vector_model.train()

        batch_loss_train = 0 
        batch_loss_dev = 0
    

        for iteration, batch in enumerate(pairs_batch_train):
            optimizer.zero_grad()
            train_loss = 0
            
            if use_trn == True:
                pad_input_seqs, input_seq_lengths, pad_trn_seqs, label_seqs = batch
                pad_input_seqs, pad_trn_seqs, label_seqs = pad_input_seqs.to(device), pad_trn_seqs.to(device), label_seqs.to(device)

                output = x_vector_model(pad_input_seqs, use_trn, trn_seqs=pad_trn_seqs)
                train_loss = criterion(output, label_seqs)
            else: 
                pad_input_seqs, input_seq_lengths, label_seqs = batch
                pad_input_seqs, label_seqs = pad_input_seqs.to(device), label_seqs.to(device)
                
                output = x_vector_model(pad_input_seqs, use_trn)
                train_loss = criterion(output, label_seqs)

            train_loss.backward()
            batch_loss_train += train_loss.detach().item() * output.size()[0]
            optimizer.step()
            
            
        # VALIDATION
        with torch.no_grad():
            x_vector_model.eval()

            for iteration, batch in enumerate(pairs_batch_dev):
                dev_loss = 0
                
                if use_trn == True:
                    pad_input_seqs, input_seq_lengths, pad_trn_seqs, label_seqs = batch
                    pad_input_seqs, pad_trn_seqs, label_seqs = pad_input_seqs.to(device), pad_trn_seqs.to(device), label_seqs.to(device)
                    
                    output = x_vector_model(pad_input_seqs, use_trn, trn_seqs=pad_trn_seqs)
                    dev_loss = criterion(output, label_seqs)
                else:    
                    pad_input_seqs, input_seq_lengths, label_seqs = batch
                    pad_input_seqs, label_seqs = pad_input_seqs.to(device), label_seqs.to(device)
                    
                    output = x_vector_model(pad_input_seqs, use_trn)
                    dev_loss = criterion(output, label_seqs)

                batch_loss_dev += dev_loss.item() * output.size()[0]
               

        # calculate accuracy
        true_topic = []
        predicted_topic = []

        for l, batch in enumerate(pairs_batch_dev_acc):
            if use_trn == True:
                pad_input_seqs, input_seq_lengths, pad_trn_seqs, label_seqs = batch
                pad_input_seqs, pad_trn_seqs, label_seqs = pad_input_seqs.to(device), pad_trn_seqs.to(device), label_seqs.to(device)
                output = x_vector_model(pad_input_seqs, use_trn, trn_seqs=pad_trn_seqs)
            else:    
                pad_input_seqs, input_seq_lengths, label_seqs = batch
                pad_input_seqs, label_seqs = pad_input_seqs.to(device), label_seqs.to(device)
                
                output = x_vector_model(pad_input_seqs, use_trn)

            output = F.softmax(output, dim=-1)
            topi, topk = output.topk(1)
            
            true_topic.append(label_seqs.item())
            predicted_topic.append(topk.item())
        
        
        accuracy = accuracy_score(true_topic, predicted_topic)

        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f   acc: %.4f' % (epoch, (batch_loss_train / len(pairs_batch_train)), (batch_loss_dev / len(pairs_batch_dev)), accuracy))


        #with open('loss/model_asr_trn.txt', 'a') as f:
        #    f.write(str(epoch) + '  ' + str(batch_loss_train / len(pairs_batch_train)) + '  ' + str(batch_loss_dev / len(pairs_batch_dev)) + '  ' + str(accuracy) + '\n')
        #
        #print('saving the model...')
        #torch.save({
        #'x_vector_model': x_vector_model.state_dict(),
        #'optimizer': optimizer.state_dict(),
        #}, 'weights/model_asr_trn/state_dict_' + str(epoch) + '.pt')
