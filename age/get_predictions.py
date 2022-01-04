import torch
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def get_predictions(x_vector_model, batch_size, idx2gender, test_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')
    
    x_vector_model.eval()

    for i in range(5, 6):
        true_age = []
        predicted_age = []

        for l, batch in enumerate(test_data):
            checkpoint = torch.load('weights/model_3_seconds/state_dict_' + str(i) + '.pt', map_location=torch.device('cpu'))
            x_vector_model.load_state_dict(checkpoint['x_vector_model'])
            x_vector_model.eval()
            
            predicted_indices = []

            pad_input_seqs, input_seq_lengths, age_seqs = batch
            pad_input_seqs, age_seqs = pad_input_seqs.to(device), age_seqs.to(device)
            output = x_vector_model(pad_input_seqs)
            output = F.softmax(output, dim=-1)
            topi, topk = output.topk(1)
            
            true_age.append(age_seqs.item())
            predicted_age.append(topk.item())
        
        print('Epoch: %.d   Accuracy: %.4f' % (i, accuracy_score(true_age, predicted_age)))


def get_predictions_relaxed(x_vector_model, batch_size, idx2gender, test_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')
    
    x_vector_model.eval()

    for i in range(5, 6):
        predicted_samples = 0

        for l, batch in enumerate(test_data):
            checkpoint = torch.load('weights/model_3_seconds/state_dict_' + str(i) + '.pt', map_location=torch.device('cpu'))
            x_vector_model.load_state_dict(checkpoint['x_vector_model'])
            x_vector_model.eval()

            pad_input_seqs, input_seq_lengths, age_seqs = batch
            pad_input_seqs, age_seqs = pad_input_seqs.to(device), age_seqs.to(device)
            
            predicted_indices = []

            output = x_vector_model(pad_input_seqs)
            output = F.softmax(output, dim=-1)
            topi, topk = output.topk(3)
            topk = topk[0]
            topi = topi[0]

            # consider neighboring classes as correct
            if (age_seqs.item() == topk[0].item()) or (age_seqs.item() == topk[0].item() + 1) or (age_seqs.item() == topk[0].item() - 1):
                predicted_samples += 1
            
        relaxed_accuracy = predicted_samples / len(test_data)

        print('Epoch: %.d   Accuracy: %.4f' % (i, relaxed_accuracy))


def get_predictions_by_age(x_vector_model, batch_size, idx2gender, test_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Evaluating...')
    
    x_vector_model.eval()

    correct_1_10 = 0
    total_1_10 = 0

    correct_11_20 = 0
    total_11_20 = 0

    correct_21_30 = 0
    total_21_30 = 0
    
    correct_31_40 = 0
    total_31_40 = 0

    correct_41_50 = 0
    total_41_50 = 0

    correct_51_60 = 0
    total_51_60 = 0

    correct_61_70 = 0
    total_61_70 = 0

    correct_71_80 = 0
    total_71_80 = 0

    correct_81_90 = 0
    total_81_90 = 0

    correct_91_100 = 0
    total_91_100 = 0

    correct_101 = 0
    total_101 = 0

    for i in range(11, 12):

        for l, batch in enumerate(test_data):
            checkpoint = torch.load('weights/model_50_seconds/state_dict_' + str(i) + '.pt', map_location=torch.device('cpu'))
            x_vector_model.load_state_dict(checkpoint['x_vector_model'])
            x_vector_model.eval()
            
            predicted_indices = []

            pad_input_seqs, input_seq_lengths, age_seqs = batch
            pad_input_seqs, age_seqs = pad_input_seqs.to(device), age_seqs.to(device)
            output = x_vector_model(pad_input_seqs)
            output = F.softmax(output, dim=-1)
            topi, topk = output.topk(1)
            
            true_age = age_seqs.item()
            predicted_age = topk.item()

            if true_age == 0:
                total_1_10 += 1
            
            if true_age == 1:
                total_11_20 += 1

            if true_age == 2:
                total_21_30 += 1

            if true_age == 3:
                total_31_40 += 1

            if true_age == 4:
                total_41_50 += 1

            if true_age == 5:
                total_51_60 += 1

            if true_age == 6:
                total_61_70 += 1

            if true_age == 7:
                total_71_80 += 1

            if true_age == 8:
                total_81_90 += 1

            if true_age == 9:
                total_91_100 += 1
            
            if true_age == 10:
                total_101 += 1


            if predicted_age == 0 and predicted_age == true_age:
                correct_1_10 += 1

            if predicted_age == 1 and predicted_age == true_age:
                correct_11_20 += 1

            if predicted_age == 2 and predicted_age == true_age:
                correct_21_30 += 1

            if predicted_age == 3 and predicted_age == true_age:
                correct_31_40 += 1

            if predicted_age == 4 and predicted_age == true_age:
                correct_41_50 += 1

            if predicted_age == 5 and predicted_age == true_age:
                correct_51_60 += 1

            if predicted_age == 6 and predicted_age == true_age:
                correct_61_70 += 1

            if predicted_age == 7 and predicted_age == true_age:
                correct_71_80 += 1

            if predicted_age == 8 and predicted_age == true_age:
                correct_81_90 += 1

            if predicted_age == 9 and predicted_age == true_age:
                correct_91_100 += 1

            if predicted_age == 10 and predicted_age == true_age:
                correct_101 += 1


        if total_1_10 >= 1:
            print('total 1-10: %.d   accuracy: %.4f' % (total_1_10, correct_1_10 / total_1_10))
        if total_11_20 >= 1:
            print('total 11-20: %.d   accuracy: %.4f' % (total_11_20, correct_11_20 / total_11_20))
        if total_21_30 >= 1:
            print('total 21-30: %.d   accuracy: %.4f' % (total_21_30, correct_21_30 / total_21_30))
        if total_31_40 >= 1:
            print('total 31-40: %.d   accuracy: %.4f' % (total_31_40, correct_31_40 / total_31_40))
        if total_41_50 >= 1:
            print('total 41-50: %.d   accuracy: %.4f' % (total_41_50, correct_41_50 / total_41_50))
        if total_51_60 >= 1:
            print('total 51-60: %.d   accuracy: %.4f' % (total_51_60, correct_51_60 / total_51_60))
        if total_61_70 >= 1:
            print('total 61-70: %.d   accuracy: %.4f' % (total_61_70, correct_61_70 / total_61_70))
        if total_71_80 >= 1:
            print('total 71-80: %.d   accuracy: %.4f' % (total_71_80, correct_71_80 / total_71_80))
        if total_81_90 >= 1:
            print('total 81-90: %.d   accuracy: %.4f' % (total_81_90, correct_81_90 / total_81_90))
        if total_91_100 >= 1:
            print('total 91-100: %.d   accuracy: %.4f' % (total_91_100, correct_91_100 / total_91_100))
        if total_101 >= 1:
            print('total 101+: %.d   accuracy: %.4f' % (total_101, correct_101 / total_101))


