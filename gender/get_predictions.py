import torch
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def get_predictions(x_vector_model, batch_size, idx2gender, test_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')
    
    x_vector_model.eval()

    for i in range(10, 11):
        true_gender = []
        predicted_gender = []

        for l, batch in enumerate(test_data):
            checkpoint = torch.load('weights/model_50_seconds/state_dict_' + str(i) + '.pt', map_location=torch.device('cpu'))
            x_vector_model.load_state_dict(checkpoint['x_vector_model'])
            x_vector_model.eval()
            
            predicted_indices = []

            pad_input_seqs, input_seq_lengths, gender_seqs = batch
            pad_input_seqs, gender_seqs = pad_input_seqs.to(device), gender_seqs.to(device)
            output = x_vector_model(pad_input_seqs)
            output = F.softmax(output, dim=-1)
            topi, topk = output.topk(1)
            
            true_gender.append(gender_seqs.item())
            predicted_gender.append(topk.item())
        
        print('Epoch: %.d   Accuracy: %.4f' % (i, accuracy_score(true_gender, predicted_gender)))



def get_predictions_by_gender(x_vector_model, batch_size, idx2gender, test_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')
    
    x_vector_model.eval()

    correct_male = 0
    total_male = 0
    correct_female = 0
    total_female = 0

    for i in range(10, 11):
        true_gender = []
        predicted_gender = []

        for l, batch in enumerate(test_data):
            checkpoint = torch.load('weights/model_50_seconds/state_dict_' + str(i) + '.pt', map_location=torch.device('cpu'))
            x_vector_model.load_state_dict(checkpoint['x_vector_model'])
            x_vector_model.eval()
            
            predicted_indices = []

            pad_input_seqs, input_seq_lengths, gender_seqs = batch
            pad_input_seqs, gender_seqs = pad_input_seqs.to(device), gender_seqs.to(device)
            output = x_vector_model(pad_input_seqs)
            output = F.softmax(output, dim=-1)
            topi, topk = output.topk(1)
            
            true_gender = gender_seqs.item()
            predicted_gender = topk.item()

            if true_gender == 0:
                total_male += 1
            elif true_gender == 1:
                total_female += 1

            if predicted_gender == 0 and predicted_gender == true_gender:
                correct_male += 1
            elif predicted_gender == 1 and predicted_gender == true_gender:
                correct_female += 1
            

        print('Total males: %.d   Accuracy: %.4f' % (total_male, correct_male / total_male))
        print('Total females: %.d   Accuracy: %.4f' % (total_female, correct_female / total_female))

