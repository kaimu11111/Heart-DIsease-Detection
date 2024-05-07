import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import random
import os
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import itertools
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.nn import functional as F
from sklearn.calibration import calibration_curve
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from statkit.decision import NetBenefitDisplay
from statistics import mean
from calibrated_metrics import average_precision, calibrated_precision
from sklearn.metrics import make_scorer

def custom_scoring_function(y_test, probs):
    return average_precision(y_test, probs, pos_label=1, sample_weight=None, pi0=0.5)

# Make a scorer from the custom scoring function
custom_scorer = make_scorer(custom_scoring_function)

def lr_evaluation(X_train, y_train, X_test, y_test):
    model = LogisticRegression(random_state=0, penalty=None, max_iter = 1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    
    training_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    sensitivity = sensitivity_score(y_test, pred, average='binary')
    specificity = specificity_score(y_test, pred, average='binary')
    
    precision = precision_score(y_test, pred, average='binary')
    calibrated_prec = calibrated_precision(y_test, pred, 0.5)
    recall = recall_score(y_test, pred, average='binary')
    f1 = f1_score(y_test, pred, average='binary')
    calibrated_f1 = 2*calibrated_prec*recall / (calibrated_prec + recall)
    
    fpr, tpr, threshold = roc_curve(y_test, prob)
    auroc = roc_auc_score(y_test, prob)
    auprc = average_precision_score(y_test, prob)
    calibrated_auprc = average_precision(y_test, prob, pos_label=1, sample_weight=None, pi0=0.5)

#     plt.plot(fpr, tpr, label="ROC")
#     plt.plot(recall, precision, label="PRC")
#     plt.legend(loc='lower right')
    
    return prob, test_accuracy, sensitivity, specificity, precision, calibrated_prec, recall, f1,calibrated_f1, auroc, auprc, calibrated_auprc
    
    
def lr_L2_evaluation(X_train, y_train, X_test, y_test):
    
    pipe_model = LogisticRegression(random_state=0,max_iter = 1000)
    param_grid = {"C":np.logspace(-3,3,7), "penalty":["l2"]}
    model = GridSearchCV(pipe_model, param_grid, cv = 10, scoring = custom_scorer)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    
    training_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    sensitivity = sensitivity_score(y_test, pred, average='binary')
    specificity = specificity_score(y_test, pred, average='binary')
    
    precision = precision_score(y_test, pred, average='binary')
    calibrated_prec = calibrated_precision(y_test, pred, 0.5)
    recall = recall_score(y_test, pred, average='binary')
    f1 = f1_score(y_test, pred, average='binary')
    calibrated_f1 = 2*calibrated_prec*recall / (calibrated_prec + recall)
    
    fpr, tpr, threshold = roc_curve(y_test, prob)
    auroc = roc_auc_score(y_test, prob)
    auprc = average_precision_score(y_test, prob)
    calibrated_auprc = average_precision(y_test, prob, pos_label=1, sample_weight=None, pi0=0.5)

#     plt.plot(fpr, tpr, label="ROC")
#     plt.plot(recall, precision, label="PRC")
#     plt.legend(loc='lower right')
    
    return prob, test_accuracy, sensitivity, specificity, precision, calibrated_prec, recall, f1,calibrated_f1, auroc, auprc, calibrated_auprc
    

def random_forest_evaluation(X_train, y_train, X_test, y_test):
    
    pipe_model = RandomForestClassifier(random_state=0)
    
    param_grid = { 
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [5, 8, 10],
        'criterion' :['gini', 'entropy']
    }
    
    model = GridSearchCV(estimator=pipe_model, param_grid=param_grid, cv = 10, scoring = custom_scorer)
    model.fit(X_train, y_train)
    print('Best hyperparameters', str(model.best_params_))
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    
    training_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    sensitivity = sensitivity_score(y_test, pred, average='binary')
    specificity = specificity_score(y_test, pred, average='binary')
    
    precision = precision_score(y_test, pred, average='binary')
    calibrated_prec = calibrated_precision(y_test, pred, 0.5)
    recall = recall_score(y_test, pred, average='binary')
    f1 = f1_score(y_test, pred, average='binary')
    calibrated_f1 = 2*calibrated_prec*recall / (calibrated_prec + recall)
    
    fpr, tpr, threshold = roc_curve(y_test, prob)
    auroc = roc_auc_score(y_test, prob)
    auprc = average_precision_score(y_test, prob)
    calibrated_auprc = average_precision(y_test, prob, pos_label=1, sample_weight=None, pi0=0.5)

#     plt.plot(fpr, tpr, label="ROC")
#     plt.plot(recall, precision, label="PRC")
#     plt.legend(loc='lower right')
    
    return prob, test_accuracy, sensitivity, specificity, precision, calibrated_prec, recall, f1,calibrated_f1, auroc, auprc, calibrated_auprc, model.best_params_
    
def random_forest_evaluation_2(X_train, y_train, X_test, y_test, best_params):
    
    model = RandomForestClassifier(random_state=0)
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    
    training_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    sensitivity = sensitivity_score(y_test, pred, average='binary')
    specificity = specificity_score(y_test, pred, average='binary')
    
    precision = precision_score(y_test, pred, average='binary')
    calibrated_prec = calibrated_precision(y_test, pred, 0.5)
    recall = recall_score(y_test, pred, average='binary')
    f1 = f1_score(y_test, pred, average='binary')
    calibrated_f1 = 2*calibrated_prec*recall / (calibrated_prec + recall)
    
    fpr, tpr, threshold = roc_curve(y_test, prob)
    auroc = roc_auc_score(y_test, prob)
    auprc = average_precision_score(y_test, prob)
    calibrated_auprc = average_precision(y_test, prob, pos_label=1, sample_weight=None, pi0=0.5)

#     plt.plot(fpr, tpr, label="ROC")
#     plt.plot(recall, precision, label="PRC")
#     plt.legend(loc='lower right')
    
    return prob, test_accuracy, sensitivity, specificity, precision, calibrated_prec, recall, f1,calibrated_f1, auroc, auprc, calibrated_auprc

def lr_simulation(num, X_train, y_train, X_test, y_test):
    
    prob_list = []
    test_accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    calibrated_precision_list = []
    recall_list = []
    f1_score_list = []
    calibrated_fl_score_list = []
    auroc_list = []
    auprc_list = []
    calibrated_auprc_list = []    
    
    for i in range(num):
        np.random.seed(i)
        
        prob, test_accuracy, sensitivity, specificity, precision, calibrated_prec, recall, f1_score, calibrated_f1, auroc, auprc, calibrated_auprc = lr_evaluation(X_train, y_train, X_test, y_test)
        
        prob_list.append(prob)
        test_accuracy_list.append(test_accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        calibrated_precision_list.append(calibrated_prec)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        calibrated_fl_score_list.append(calibrated_f1)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
        calibrated_auprc_list.append(calibrated_auprc)
        
    
    mean_prob_list = np.average(np.array(prob_list), axis = 0)
    mean_accuracy = mean(test_accuracy_list)
    mean_sensitivity= mean(sensitivity_list)
    mean_specificity = mean(specificity_list)
    mean_precision = mean(precision_list)
    mean_calibrated_precision = mean(calibrated_precision_list)
    mean_recall = mean(recall_list)
    mean_fl_score = mean(f1_score_list)
    mean_calibrated_f1 = mean(calibrated_fl_score_list)
    mean_auroc = mean(auroc_list)
    mean_auprc = mean(auprc_list)
    mean_calibrated_auprc = mean(calibrated_auprc_list)
    
    print('Test accuracy: %.3f' % mean_accuracy)
    print('Sensitivity: %.3f' % mean_sensitivity)
    print('Specificity: %.3f' % mean_specificity)
    print('Precision: %.3f' % mean_precision)
    print('Calibrated Precision: %.3f' % mean_calibrated_precision)
    print('Recall: %.3f' % mean_recall)
    print('F1 Score: %.3f' % mean_fl_score)
#     print('Calibrated F1 Score: %.3f' % mean_calibrated_f1)
    print('AUROC: %.3f' % mean_auroc) 
    print('AUPRC: %.3f' % mean_auprc)
    print('Calibrated AUPRC: %.3f' % mean_calibrated_auprc)
    
    print(mean_accuracy)
    print(mean_sensitivity)
    print(mean_specificity)
    print(mean_precision)
    print(mean_calibrated_precision)
    print(mean_recall)
    print(mean_fl_score)
    print(mean_auroc) 
    print(mean_auprc)
    print(mean_calibrated_auprc)

    
    
    return mean_prob_list

def lr_L2_simulation(num, X_train, y_train, X_test, y_test):
    
    prob_list = []
    test_accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    calibrated_precision_list = []
    recall_list = []
    f1_score_list = []
    calibrated_fl_score_list = []
    auroc_list = []
    auprc_list = []
    calibrated_auprc_list = [] 
    
    for i in range(num):
        np.random.seed(i)   
        
        prob, test_accuracy, sensitivity, specificity, precision, calibrated_prec, recall, f1_score, calibrated_f1, auroc, auprc, calibrated_auprc = lr_L2_evaluation(X_train, y_train, X_test, y_test)
        
        prob_list.append(prob)
        test_accuracy_list.append(test_accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        calibrated_precision_list.append(calibrated_prec)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        calibrated_fl_score_list.append(calibrated_f1)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
        calibrated_auprc_list.append(calibrated_auprc)
        
    
    mean_prob_list = np.average(np.array(prob_list), axis = 0)
    mean_accuracy = mean(test_accuracy_list)
    mean_sensitivity= mean(sensitivity_list)
    mean_specificity = mean(specificity_list)
    mean_precision = mean(precision_list)
    mean_calibrated_precision = mean(calibrated_precision_list)
    mean_recall = mean(recall_list)
    mean_fl_score = mean(f1_score_list)
    mean_calibrated_f1 = mean(calibrated_fl_score_list)
    mean_auroc = mean(auroc_list)
    mean_auprc = mean(auprc_list)
    mean_calibrated_auprc = mean(calibrated_auprc_list)
    
    print('Test accuracy: %.3f' % mean_accuracy)
    print('Sensitivity: %.3f' % mean_sensitivity)
    print('Specificity: %.3f' % mean_specificity)
    print('Precision: %.3f' % mean_precision)
    print('Calibrated Precision: %.3f' % mean_calibrated_precision)
    print('Recall: %.3f' % mean_recall)
    print('F1 Score: %.3f' % mean_fl_score)
#     print('Calibrated F1 Score: %.3f' % mean_calibrated_f1)
    print('AUROC: %.3f' % mean_auroc) 
    print('AUPRC: %.3f' % mean_auprc)
    print('Calibrated AUPRC: %.3f' % mean_calibrated_auprc)
    
    print(mean_accuracy)
    print(mean_sensitivity)
    print(mean_specificity)
    print(mean_precision)
    print(mean_calibrated_precision)
    print(mean_recall)
    print(mean_fl_score)
    print(mean_auroc) 
    print(mean_auprc)
    print(mean_calibrated_auprc)
    
    return mean_prob_list

def rf_simulation(num, X_train, y_train, X_test, y_test):
    
    prob_list = []
    test_accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    calibrated_precision_list = []
    recall_list = []
    f1_score_list = []
    calibrated_fl_score_list = []
    auroc_list = []
    auprc_list = []
    calibrated_auprc_list = []   
    
    for i in range(num):
        np.random.seed(i)
        
        if i == 0:
            prob, test_accuracy, sensitivity, specificity, precision, calibrated_prec, recall, f1_score, calibrated_f1, auroc, auprc, calibrated_auprc, best_params = random_forest_evaluation(X_train, y_train, X_test, y_test)
        else:
            prob, test_accuracy, sensitivity, specificity, precision, calibrated_prec, recall, f1_score, calibrated_f1, auroc, auprc, calibrated_auprc = random_forest_evaluation_2(X_train, y_train, X_test, y_test, best_params)
            
        
        prob_list.append(prob)
        test_accuracy_list.append(test_accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        calibrated_precision_list.append(calibrated_prec)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        calibrated_fl_score_list.append(calibrated_f1)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
        calibrated_auprc_list.append(calibrated_auprc)
        
    
    mean_prob_list = np.average(np.array(prob_list), axis = 0)
    mean_accuracy = mean(test_accuracy_list)
    mean_sensitivity= mean(sensitivity_list)
    mean_specificity = mean(specificity_list)
    mean_precision = mean(precision_list)
    mean_calibrated_precision = mean(calibrated_precision_list)
    mean_recall = mean(recall_list)
    mean_fl_score = mean(f1_score_list)
    mean_calibrated_f1 = mean(calibrated_fl_score_list)
    mean_auroc = mean(auroc_list)
    mean_auprc = mean(auprc_list)
    mean_calibrated_auprc = mean(calibrated_auprc_list)
    
    print('Test accuracy: %.3f' % mean_accuracy)
    print('Sensitivity: %.3f' % mean_sensitivity)
    print('Specificity: %.3f' % mean_specificity)
    print('Precision: %.3f' % mean_precision)
    print('Calibrated Precision: %.3f' % mean_calibrated_precision)
    print('Recall: %.3f' % mean_recall)
    print('F1 Score: %.3f' % mean_fl_score)
#     print('Calibrated F1 Score: %.3f' % mean_calibrated_f1)
    print('AUROC: %.3f' % mean_auroc) 
    print('AUPRC: %.3f' % mean_auprc)
    print('Calibrated AUPRC: %.3f' % mean_calibrated_auprc)
    
    print(mean_accuracy)
    print(mean_sensitivity)
    print(mean_specificity)
    print(mean_precision)
    print(mean_calibrated_precision)
    print(mean_recall)
    print(mean_fl_score)
    print(mean_auroc) 
    print(mean_auprc)
    print(mean_calibrated_auprc)
    
    return mean_prob_list

def check_auprc(model, loader):
    model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss()
    correct_output = 0
    total_output = 0
    probs = []
    y_test = []
    loss_list = []
    val_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            
            y = y.flatten().long()
            score = model(x)
            loss = loss_criterion(score, y)
            
            _, predictions = score.max(1)
            m = nn.Softmax(dim=1)
            val_prob = m(score).detach().cpu().numpy()[:, 1]
            probs.append(val_prob)
            y_test.append(y.detach().cpu().numpy())
            correct_output += (y==predictions).sum()
            total_output += predictions.shape[0]
            
            log_loss = loss.item()
            val_loss += log_loss
    age_loss = val_loss/(i+1)
    probs = np.concatenate(probs, axis=0)
    y_test = np.concatenate(y_test, axis=0)
#     auprc = average_precision_score(y_test, probs)
    auprc = average_precision(y_test, probs, pos_label=1, sample_weight=None,pi0=0.5)
#     print(f"out of {total_output} , total correct: {correct_output} with an calibrated auprc of {auprc}")
    accuracy = float(correct_output/total_output)*100
    return auprc, age_loss

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def Neural_Network_Evaluation(X_train, y_train, X_val, y_val, X_test, y_test, epochs):

    trainset_x = torch.from_numpy(X_train.values).float() 
    trainset_y = torch.from_numpy(y_train.values.ravel()).float() 
    trainset_y = trainset_y.unsqueeze(1)
    
    valset_x = torch.from_numpy(X_val.values).float() 
    valset_y = torch.from_numpy(y_val.values.ravel()).float()
    valset_y = valset_y.unsqueeze(1)
    
    testset_x = torch.from_numpy(X_test.values).float()
    testset_y = torch.from_numpy(y_test.values.ravel()).float()
    testset_y = testset_y.unsqueeze(1)
    
    trainset = TensorDataset(trainset_x, trainset_y)
    valset = TensorDataset(valset_x, valset_y)
    testset = TensorDataset(testset_x, testset_y)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    valloader = DataLoader(valset, batch_size=128, shuffle=False)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)
    
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim=100, num_layers=10, output_dim=2):
            super().__init__()
            bias = False
            layers = [nn.Linear(input_dim, hidden_dim, bias=bias)]
            for _ in range(num_layers-2):
                layers.append(nn.BatchNorm1d(num_features = hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if output_dim == 2:
                layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
            else:
                layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

            # layers = [nn.Linear(input_dim, 1, bias=bias)]
            # layers.append(nn.Sigmoid())


            self.net = nn.Sequential(*layers)
        def forward(self, X):
            return self.net(X)
    
    model = MLP(trainset_x.shape[1])
    learning_rate = 0.001
    best_accuracy = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    #forward loop
    train_loss_list = []
    val_loss_list = []
    auprc_list = []
    best_auprc = 0
    best_model = None
    
    for i in range(epochs):
        model.train()
        epoch_loss = 0
        for j, (xb, yb) in enumerate(trainloader):
            model.train()
            yb = yb.flatten().long()
            optimizer.zero_grad()
            #calculate output
            output = model(xb)
            #calculate loss
            loss = loss_fn(output, yb)

            #backprop
            loss.backward()
            optimizer.step()
            
            log_loss = loss.item()
            epoch_loss += log_loss

        age_loss = epoch_loss/(j+1)
        train_loss_list.append(age_loss)
        auprc, val_loss = check_auprc(model, valloader)
        if auprc > best_auprc:
            best_auprc = auprc
            value_loss = val_loss
            value_auprc = auprc
            val_loss_list.append(value_loss)
            auprc_list.append(value_auprc)
            best_model = model.state_dict().copy()
        else:
            value_loss = val_loss_list[-1]
            value_auprc = auprc_list[-1]
            val_loss_list.append(value_loss)
            auprc_list.append(value_auprc)
            
        print(f"current loss: {value_loss} with an calibrated auprc of {value_auprc}")
        
    f1 = plt.figure(1)
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)

    f2 = plt.figure(2)
    plt.plot(auprc_list)
    
    y_pred_list = []
    y_prob_list = []
    model.eval()
    with torch.no_grad():
        for xb_test, yb_test in testloader:
            y_test_pred = model(xb_test)
            dim = nn.Softmax(dim=1)
            y_pred_tag = dim(y_test_pred).detach().cpu().numpy().round()[:, 1]
            y_test_pred = dim(y_test_pred).detach().cpu().numpy()[:, 1]
            y_prob_list.append(y_test_pred)
            y_pred_list.append(y_pred_tag)

    #Takes arrays and makes them list of list for each batch        
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_prob_list = [a.squeeze().tolist() for a in y_prob_list]
    #flattens the lists in sequence
    ytest_pred = np.array(list(itertools.chain.from_iterable(y_pred_list)))
    ytest_prob = np.array(list(itertools.chain.from_iterable(y_prob_list)))
    y_true_test = y_test.values.ravel()
    #conf_matrix = confusion_matrix(y_true_test ,ytest_pred)
    
    test_accuracy = accuracy_score(y_true_test ,ytest_pred)
    sensitivity = sensitivity_score(y_true_test ,ytest_pred, average='binary')
    specificity = specificity_score(y_true_test ,ytest_pred, average='binary')
    
    precision = precision_score(y_true_test, ytest_pred, average='binary')
    calibrated_prec = calibrated_precision(y_true_test, ytest_pred, 0.5)
    recall = recall_score(y_true_test, ytest_pred, average='binary')
    f1 = f1_score(y_true_test, ytest_pred, average='binary')
    calibrated_f1 = 2*calibrated_prec*recall / (calibrated_prec + recall)

    fpr, tpr, threshold = roc_curve(y_true_test, ytest_prob)
    auroc = roc_auc_score(y_true_test, ytest_prob)
#     precision, recall, threshold = precision_recall_curve(y_true_test, ytest_prob)
    auprc = average_precision_score(y_true_test, ytest_prob)
    calibrated_auprc = average_precision(y_true_test, ytest_prob, pos_label=1, sample_weight=None, pi0=0.5)

#     f3 = plt.figure(3)
#     plt.plot(fpr, tpr, label="ROC")
#     plt.plot(recall, precision, label="PRC")
#     plt.legend(loc='lower right')

    print("auprc : ", auprc)
    print("calibrated_auprc : ", calibrated_auprc)
    
    return ytest_prob, test_accuracy, sensitivity, specificity, precision, calibrated_prec, recall, f1, calibrated_f1, auroc, auprc, calibrated_auprc

def NN_simulation(num, X_train, y_train, X_val, y_val, X_test, y_test, epochs):
    
    prob_list = []
    test_accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    calibrated_precision_list = []
    recall_list = []
    f1_score_list = []
    calibrated_fl_score_list = []
    auroc_list = []
    auprc_list = []
    calibrated_auprc_list = []
    
    for i in range(num):
        set_seed(i)
       
        prob, test_accuracy, sensitivity, specificity, precision, calibrated_prec, recall, f1_score, calibrated_f1, auroc, auprc, calibrated_auprc = Neural_Network_Evaluation(X_train, y_train, X_val, y_val, X_test, y_test, epochs = 50)
        
        prob_list.append(prob)
        test_accuracy_list.append(test_accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        calibrated_precision_list.append(calibrated_prec)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        calibrated_fl_score_list.append(calibrated_f1)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
        calibrated_auprc_list.append(calibrated_auprc)
    
    mean_prob_list = np.average(np.array(prob_list), axis = 0)
    mean_accuracy = mean(test_accuracy_list)
    mean_sensitivity= mean(sensitivity_list)
    mean_specificity = mean(specificity_list)
    mean_precision = mean(precision_list)
    mean_calibrated_precision = mean(calibrated_precision_list)
    mean_recall = mean(recall_list)
    mean_fl_score = mean(f1_score_list)
    mean_calibrated_f1 = mean(calibrated_fl_score_list)
    mean_auroc = mean(auroc_list)
    mean_auprc = mean(auprc_list)
    mean_calibrated_auprc = mean(calibrated_auprc_list)
    
    print('Test accuracy: %.3f' % mean_accuracy)
    print('Sensitivity: %.3f' % mean_sensitivity)
    print('Specificity: %.3f' % mean_specificity)
    print('Precision: %.3f' % mean_precision)
    print('Calibrated Precision: %.3f' % mean_calibrated_precision)
    print('Recall: %.3f' % mean_recall)
    print('F1 Score: %.3f' % mean_fl_score)
#     print('Calibrated F1 Score: %.3f' % mean_calibrated_f1)
    print('AUROC: %.3f' % mean_auroc) 
    print('AUPRC: %.3f' % mean_auprc)
    print('Calibrated AUPRC: %.3f' % mean_calibrated_auprc)
    
    print(mean_accuracy)
    print(mean_sensitivity)
    print(mean_specificity)
    print(mean_precision)
    print(mean_calibrated_precision)
    print(mean_recall)
    print(mean_fl_score)
    print(mean_auroc) 
    print(mean_auprc)
    print(mean_calibrated_auprc)
    
    return mean_prob_list