import torch
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def get_predictions(x_vector_model, batch_size, test_data, use_trn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')
    
    x_vector_model.eval()

    for i in range(8, 9):
        true_label = []
        predicted_label = []

        for l, batch in enumerate(test_data):
            checkpoint = torch.load('weights/model_audio_subset/state_dict_' + str(i) + '.pt', map_location=torch.device('cpu'))
            x_vector_model.load_state_dict(checkpoint['x_vector_model'])
            x_vector_model.eval()
            
            predicted_indices = []
            
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
            
            true_label.append(label_seqs.item())
            predicted_label.append(topk.item())
            
        np.save('output/true.npy', true_label)
        np.save('output/predicted.npy', predicted_label)
        
        print('Epoch: %.d   Accuracy: %.4f' % (i, accuracy_score(true_label, predicted_label)))



def get_predictions_by_dialect(x_vector_model, batch_size, test_data, use_trn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')
    
    x_vector_model.eval()
    
    correct_0 = 0
    total_0 = 0

    correct_1 = 0
    total_1 = 0

    correct_2 = 0
    total_2 = 0
    
    correct_3 = 0
    total_3 = 0

    correct_4 = 0
    total_4 = 0

    correct_5 = 0
    total_5 = 0

    correct_6 = 0
    total_6 = 0

    correct_7 = 0
    total_7 = 0


    for i in range(4, 5):
        true_label = []
        predicted_label = []

        for l, batch in enumerate(test_data):
            checkpoint = torch.load('weights/model_audio_trn/state_dict_' + str(i) + '.pt', map_location=torch.device('cpu'))
            x_vector_model.load_state_dict(checkpoint['x_vector_model'])
            x_vector_model.eval()
            
            predicted_indices = []
            
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
            
            true_label = label_seqs.item()
            predicted_label = topk.item()


            if true_label == 0:
                total_0 += 1
            
            if true_label == 1:
                total_1 += 1

            if true_label == 2:
                total_2 += 1

            if true_label == 3:
                total_3 += 1

            if true_label == 4:
                total_4 += 1

            if true_label == 5:
                total_5 += 1
  
            if true_label == 6:
                total_6 += 1
   
            if true_label == 7:
                total_7 += 1
            
      
            if predicted_label == 0 and predicted_label == true_label:
                correct_0 += 1
            
            if predicted_label == 1 and predicted_label == true_label:
                correct_1 += 1

            if predicted_label == 2 and predicted_label == true_label:
                correct_2 += 1

            if predicted_label == 3 and predicted_label == true_label:
                correct_3 += 1

            if predicted_label == 4 and predicted_label == true_label:
                correct_4 += 1

            if predicted_label == 5 and predicted_label == true_label:
                correct_5 += 1

            if predicted_label == 6 and predicted_label == true_label:
                correct_6 += 1

            if predicted_label == 7 and predicted_label == true_label:
                correct_7 += 1

            
        if total_0 != 0: 
            print('total 0: %.d   accuracy: %.4f' % (total_0, correct_0 / total_0))

        if total_1 != 0: 
            print('total 1: %.d   accuracy: %.4f' % (total_1, correct_1 / total_1))
        
        if total_2 != 0:
            print('total 2: %.d   accuracy: %.4f' % (total_2, correct_2 / total_2))

        if total_3 != 0: 
            print('total 3: %.d   accuracy: %.4f' % (total_3, correct_3 / total_3))
        
        if total_4 != 0: 
            print('total 4: %.d   accuracy: %.4f' % (total_4, correct_4 / total_4))

        if total_5 != 0: 
            print('total 5: %.d   accuracy: %.4f' % (total_5, correct_5 / total_5))

        if total_6 != 0: 
            print('total 6: %.d   accuracy: %.4f' % (total_6, correct_6 / total_6))
         
        if total_7 != 0: 
            print('total 7: %.d   accuracy: %.4f' % (total_7, correct_7 / total_7))


