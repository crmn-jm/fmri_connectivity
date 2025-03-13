#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################### SEEDS
# For custom operators
import random
random.seed(0)
# To seed the RNG for all devices
import torch
torch.manual_seed(0)
# To seed the global NumPy RNG
import numpy as np
np.random.seed(0)

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import make_grid
# import matplotlib.pyplot as puet
import math
import os
import numpy as np
import scipy.io as sio
from scipy.special import comb
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder 
import pickle
from patchify import patchify
import nibabel as nib
from sklearn.metrics import hinge_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import h5py

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torch.autograd import Variable
import torchvision.models as models


#%%
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

server = False
aleatoridad = False
balanceo = False
augmentation = False
early_stop = False
LOO = False
kfold_CV = True
RUB = False
CVRUB = False
FP_test = False
FES =  10 # PCA:0 PLS:1 2: no reduction
Npca = 3 # PCA scores
Npls = 3 # PLS scores

if server==False:
    os.chdir('/main_path/')

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def perf_measure(y_true,y_pred):
    cm = confusion_matrix(y_true,y_pred)
    confusion_matrix(y_true,y_pred) 
    #####from confusion matrix calculate accuracy
    total=sum(sum(cm))
    acc=(cm[0,0]+cm[1,1])/total
    sens = cm[0,0]/(cm[0,0]+cm[0,1])
    spec = cm[1,1]/(cm[1,0]+cm[1,1])
    prec = cm[0,0]/(cm[0,0]+cm[1,0])

    return acc,sens,spec, prec 

def perf_measure_inversa(y_true,y_pred):
    cm = confusion_matrix(y_true,y_pred)
    confusion_matrix(y_true,y_pred) 
    #####from confusion matrix calculate accuracy
    total=sum(sum(cm))
    acc=(cm[0,0]+cm[1,1])/total
    sens = cm[1,1]/(cm[1,1]+cm[1,0])
    spec = cm[0,0]/(cm[0,1]+cm[0,0])
    prec = cm[1,1]/(cm[1,1]+cm[0,1])
    #print(classification_report(y_true,y_pred))
    return acc,sens,spec, prec

def normalize_imageset_simple(imset):
    n,w,l=imset.shape[0],imset.shape[1],imset.shape[2]
    imset=np.reshape(imset,[n,w*l])
    maximos = np.max(imset,axis=1)
    for i in range(maximos.shape[0]):
        imset[i,:] = imset[i,:]/maximos[i]
        
    imset=imset.reshape(n,w,l)
    return imset.astype(np.float32)


def bound_bayes(n,alpha,dropout_rate,clf, acc):
    #  Inputs:
    #    n:       Sample size
    #    alpha:   Significance level  
    #    clf:     classification model   
    
    lambda_para = np.arange(1/2+0.1,10.1,0.1) # parameter upper bounding the root
    a = 1./(1-1./(2*lambda_para))
    k=np.arange(1,len(a)+1,1)
    Lmax=1 # maximum error
    # Bound: L=Lemp+Bound is the worst case, Bound=(a-1)*Lemp+a*B
    # where B= (Lambda*Lmax)/N * ( (1-dropout)/2 ||theta||^2 + ln (1/alpha) )
    
    Theta=np.concatenate( (clf.coef_.reshape(-1) , clf.intercept_))
    #dropout_rate=0.95 # selected values 0 (no dropout) or 0.5 (middle effect), 0.95 strong dropout
    
    boundB  = np.nanmin( (a-1)*(1-acc) +  a*(lambda_para*Lmax/n)*( (1-dropout_rate)/2 * np.linalg.norm(Theta,2)**2 + np.log(k/alpha)  ) )
    # minimum in lambda    
    
    return round(boundB,4)

#%%  
print(" -------- Loading data ---------")

stack1 = h5py.File('path_connectivity_matrix','r')
data = np.asarray(stack1['matvec2D_red'])
data = np.moveaxis(data,-1,0)
if len(data.shape) > 2:
    data = np.moveaxis(data,1,2)

datalabels = np.zeros(data.shape[0])
datalabels[29:] = 1 


if balanceo:
    data_clase0 = data[datalabels==0]
    data_clase1 = data[datalabels==1]
    ids = np.random.permutation(data_clase0.shape[0])
    data_clase1 = data_clase1[ids[0:data_clase0.shape[0]]]
    datalabels = np.hstack((np.zeros(data_clase0.shape[0]),np.ones(data_clase1.shape[0])))
    data = np.vstack((data_clase0,data_clase1))

    save_obj(ids, 'idx_balance')
    ids=load_obj('idx_balance')
    
if aleatoridad:
    orden = np.random.permutation(data.shape[0])
    data = data[orden]
    datalabels = datalabels[orden]
    
    save_obj(orden, 'sorted')
    orden=load_obj('sorted')


scaler = StandardScaler()

#%% K-fold Classification

if kfold_CV == True:
    print("----------- k-Fold CV ------------")   
    print("----------------------------------")  

    n_folds = 10
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
            
    train_id=[]
    test_id=[]
    val_id = []
    for train_index, test_index in skf.split(data, datalabels): 
        train_id.append(train_index)
        test_id.append(test_index)
    
    train_idx=train_id
    test_idx=test_id
        
    acc_eval_tt = []
    acc_eval_tr = []
    lr_auc_eval_tt = []
    lr_auc_eval_tr = [] 
    perf_tt_all = []       
    history_results = {}
    coeficientes = []
    for s in range(0,n_folds):
        print("Training set size: ", train_idx[s].shape[0])
        print("Test set size: ", test_idx[s].shape[0])
        
        history = {}
        
        labels_arr = np.copy(datalabels)
        trainlabels_arr = labels_arr[train_idx[s]]
        testlabels_arr = labels_arr[test_idx[s]]
        clase1 = np.sum(trainlabels_arr==0)
        clase2 = np.sum(trainlabels_arr==1)
        nsamples = [clase1,clase2]

        
        data_train = data[train_idx[s]]
        data_test = data[test_idx[s]]
        if len(data.shape) > 2:
            data_train = np.reshape(data_train,[data_train.shape[0],data_train.shape[1]*data_train.shape[2]])
            data_test = np.reshape(data_test,[data_test.shape[0],data_test.shape[1]*data_test.shape[2]])
        
        
        if FES == 0:
            pca = PCA(n_components=Npca)
            pca.fit(data_train)
            X_t_train = pca.transform(data_train)
            X_t_test = pca.transform(data_test)
        elif FES == 1:
            a=1 #pls
            plsca = PLSRegression(n_components=Npls)   
            plsca.fit(data_train, trainlabels_arr.flatten()) 
            X_t_train,_ = plsca.transform(data_train, trainlabels_arr.flatten())
            X_t_test,_ = plsca.transform(data_test, testlabels_arr.flatten()) 
        else: 
            X_t_train = np.copy(data_train)
            X_t_test  = np.copy(data_test)
            
            
        X_t_train_norm=scaler.fit_transform(X_t_train)
        X_t_test_norm=scaler.transform(X_t_test)    
        
        clf = SVC(kernel='linear',class_weight='balanced',probability=True)
        clf.fit(X_t_train_norm, trainlabels_arr)
        coeficientes.append(clf.coef_)
        
        #### Results
        pred_train = clf.predict(X_t_train_norm)
        acc_final_tr = balanced_accuracy_score(trainlabels_arr,pred_train)
        acc_eval_tr.append(acc_final_tr)
        probs_train = clf.predict_proba(X_t_train_norm)
        lr_auc_tr = roc_auc_score(np.array(trainlabels_arr), np.array(probs_train[:, 1]))
        lr_auc_eval_tr.append(lr_auc_tr)
        perf_tr=perf_measure(pred_train,trainlabels_arr)
        print(f'>>> FOLD {s}: balanced accuracy in training set =  {acc_final_tr} ')
        print(classification_report(trainlabels_arr, pred_train,zero_division=0))
        
        pred_test = clf.predict(X_t_test_norm)
        acc_final_tt = balanced_accuracy_score(testlabels_arr,pred_test)
        acc_eval_tt.append(acc_final_tt)
        probs_test = clf.predict_proba(X_t_test_norm)
        lr_auc_tt = roc_auc_score(np.array(testlabels_arr), np.array(probs_test[:, 1]))
        lr_auc_eval_tt.append(lr_auc_tt)
        perf_tt=perf_measure(pred_test,testlabels_arr)
        perf_tt_all.append(perf_tt)
        print(f'>>> FOLD {s}: balanced accuracy in test set =  {acc_final_tt} ')
        print(classification_report(testlabels_arr, pred_test,zero_division=0))
        
        print(f'>>> Test accuracies so far: { acc_eval_tt}')
    
    
        ####### Save results #######
        print('>>>>>> This fold:  ')
        history['data_train'] = data_train
        history['data_train_reduced'] = X_t_train_norm
        history['predicted_train'] = pred_train
        history['labels_train'] = trainlabels_arr
        history['probs_train'] = probs_train
        history['bal_acc_train'] = acc_final_tr
        history['auc_score_train'] = lr_auc_tr
        history['perf_measure_train'] = perf_tr

        history['data_test'] = data_test
        history['data_test_reduced'] = X_t_test_norm    
        history['predicted_test'] = pred_test
        history['labels_test'] = testlabels_arr
        history['probs_test'] = probs_test
        history['bal_acc_test'] = acc_final_tt
        history['auc_score_test'] = lr_auc_tt
        history['perf_measure_test'] = perf_tt
        
        
        history_results[s]= history
        save_obj(history_results, "history_results_kfold") 
    
    perf_tt_media = np.array(perf_tt_all)
    accs_media_tr = np.array(acc_eval_tr)
    accs_media_tt = np.array(acc_eval_tt)
    lr_auc_medio_tr = np.array(lr_auc_eval_tr)
    lr_auc_medio_tt = np.array(lr_auc_eval_tt)
    coeficientes_total = np.array(coeficientes)
    coeficientes_medio = np.squeeze(np.mean(coeficientes_total,axis=0))
    
    if len(data.shape) > 2:
        coef_imagen = np.reshape(coeficientes_medio,[data.shape[1],data.shape[2]])
        plt.figure(figsize=(12, 6), dpi=80)
        plt.imshow(coef_imagen, cmap='viridis')  
        plt.colorbar()
        plt.show()
        fig, ax = plt.subplots()
        
        #sio.savemat('coef_clasificacion.mat', {'coef_imagen': coef_imagen})

    
    history_results['mean_acc_tr'] = np.mean(accs_media_tr)
    history_results['mean_acc_tt'] = np.mean(accs_media_tt)
    history_results['mean_auc_tr'] = np.mean(lr_auc_medio_tr)
    history_results['mean_auc_tt'] = np.mean(lr_auc_medio_tt)
    history_results['coeficientes'] = coeficientes_medio 
    save_obj(history_results, "history_results_kfold") 
      
    print('---------------- Final accuracy metrics ----------------')
    print(f'>>> Acc media train: {np.mean(accs_media_tr)} +- {np.std(accs_media_tr)}')
    print(f'>>> Acc media test: {np.mean(accs_media_tt)} +- {np.std(accs_media_tt)}')
    print(f'>>> ROC AUC score medio train = {np.mean(lr_auc_medio_tr)} +- {np.std(lr_auc_medio_tr)} ')   
    print(f'>>> ROC AUC score medio test = {np.mean(lr_auc_medio_tt)} +- {np.std(lr_auc_medio_tt)}') 
    
#%% Clasificación con RUB

if RUB == True:
    print("----------- RESUSTITUTION WITH UPPER BOUND CORRECTION ------------")    
    

    if FES == 0:
        d=Npca
    elif FES == 1:
        d=Npls
    else:
        d=data.shape[1]
    alpha = 0.05

    history_results = {}
    
    trainlabels_arr = np.copy(datalabels)
    clase1 = np.sum(trainlabels_arr==0)
    clase2 = np.sum(trainlabels_arr==1)
    nsamples = [clase1,clase2]
    
    data_train = np.copy(data)
    if len(data.shape) > 2:
        data_train = np.reshape(data_train,[data_train.shape[0],data_train.shape[1]*data_train.shape[2]])
    print("Training set size: ", data_train.shape[0])  
    
    if FES == 0:
        pca = PCA(n_components=Npca)
        pca.fit(data_train)
        X_t_train = pca.transform(data_train)
    elif FES == 1:
        a=1 #pls
        plsca = PLSRegression(n_components=Npls)   
        plsca.fit(data_train, trainlabels_arr.flatten()) 
        X_t_train,_ = plsca.transform(data_train, trainlabels_arr.flatten())
    else: 
        X_t_train = np.copy(data_train)

        
    X_t_train_norm=scaler.fit_transform(X_t_train)   
           
    clf = SVC(kernel='linear',class_weight='balanced',probability=True)
    clf.fit(X_t_train_norm, trainlabels_arr)
    coeficientesRUB = clf.coef_
    
    #### Results
    pred_train = clf.predict(X_t_train_norm)
    acc_tr = balanced_accuracy_score(trainlabels_arr,pred_train)
    boundVC = bound_bayes(data.shape[0],alpha,0.95,clf, acc_tr)
    acc_eval_tr = acc_tr - boundVC
    probs_train = clf.predict_proba(X_t_train_norm)
    lr_auc_eval_tr = roc_auc_score(np.array(trainlabels_arr), np.array(probs_train[:, 1]))
    perf_tr=perf_measure(pred_train,trainlabels_arr)



    ####### Save fold results #######
    print('>>>>>> Saving results...  ')
    history_results['data_train'] = data_train
    history_results['data_train_reduced'] = X_t_train_norm
    history_results['predicted_train'] = pred_train
    history_results['labels_train'] = trainlabels_arr
    history_results['probs_train'] = probs_train
    history_results['bal_acc_train'] = acc_eval_tr
    history_results['auc_score_train'] = lr_auc_eval_tr
    history_results['perf_measure_train'] = perf_tr
    history_results['perf_measure_RUB_train'] = np.array(perf_tr) - boundVC

    
    save_obj(history_results, "history_results_RUB") 
    
    if len(data.shape) > 2:
        coef_imagenRUB = np.reshape(coeficientesRUB,[data.shape[1],data.shape[2]])
        plt.figure(figsize=(12, 6), dpi=80)
        plt.imshow(coef_imagenRUB, cmap='viridis')  
        plt.colorbar()
        plt.show()
        
      
    print('---------------- Final accuracy metrics ----------------')
    print(f'>>> Acc train: {acc_tr}')
    print(f'>>> Bound RUB: {boundVC}')
    print(f'>>> Acc RUB train: {acc_eval_tr}')
    print(f'>>> ROC AUC score medio train = {lr_auc_eval_tr}')   

#%% ESTUDIO DE PERMUTACIONES $$$$$
###################################
iterations = 1000

# n = 58  # Puedes ajustar el tamaño del vector según tus necesidades
# etiquetas = [0 if i % 2 == 0 else 1 for i in range(n)]


if FP_test == True:
    print("----------- k-Fold CV ------------")   
    print("----------------------------------")  

    accs_media_tr = np.zeros([iterations,1])
    accs_media_tt = np.zeros([iterations,1])
    lr_auc_medio_tr = np.zeros([iterations,1])
    lr_auc_medio_tt = np.zeros([iterations,1]) 
    orden = np.zeros([data.shape[0],iterations]).astype(int)
    
    for i in range(iterations):
        orden[:,i] = np.random.permutation(data.shape[0])
    
    for i in range(iterations):
        datalabels = datalabels[orden[:,i]]        
        
        n_folds = 10
    
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
                
        train_id=[]
        test_id=[]
        val_id = []
        for train_index, test_index in skf.split(data, datalabels): 
            train_id.append(train_index)
            test_id.append(test_index)
        
        ### If already saved
        # train_idx=load_obj('train_idx')
        # test_idx=load_obj('test_idx')
        ### If not
        train_idx=train_id
        test_idx=test_id
            
        acc_eval_tt = []
        acc_eval_tr = []
        lr_auc_eval_tt = []
        lr_auc_eval_tr = []        

        for s in range(0,n_folds):

            
            history = {}
            
            labels_arr = np.copy(datalabels)
            trainlabels_arr = labels_arr[train_idx[s]]
            testlabels_arr = labels_arr[test_idx[s]]
            clase1 = np.sum(trainlabels_arr==0)
            clase2 = np.sum(trainlabels_arr==1)
            nsamples = [clase1,clase2]
            # normedWeights = [1 - (x / sum(nsamples)) for x in nsamples]
            
            # class_weight = {0 : (trainlabels_arr == 1).sum() / (trainlabels_arr == 0).sum(), 
            #                 1 : 1.}
            
            data_train = data[train_idx[s]]
            data_test = data[test_idx[s]]
            if len(data.shape) > 2:
                data_train = np.reshape(data_train,[data_train.shape[0],data_train.shape[1]*data_train.shape[2]])
                data_test = np.reshape(data_test,[data_test.shape[0],data_test.shape[1]*data_test.shape[2]])
            
            
            if FES == 0:
                pca = PCA(n_components=Npca)
                pca.fit(data_train)
                X_t_train = pca.transform(data_train)
                X_t_test = pca.transform(data_test)
            elif FES == 1:
                a=1 #pls
                plsca = PLSRegression(n_components=Npls)   
                plsca.fit(data_train, trainlabels_arr.flatten()) 
                X_t_train,_ = plsca.transform(data_train, trainlabels_arr.flatten())
                X_t_test,_ = plsca.transform(data_test, testlabels_arr.flatten()) 
            else: 
                X_t_train = np.copy(data_train)
                X_t_test  = np.copy(data_test)
                
                
            X_t_train_norm=scaler.fit_transform(X_t_train)
            X_t_test_norm=scaler.transform(X_t_test)    
                   
            clf = SVC(kernel='linear',class_weight='balanced',probability=True)
            clf.fit(X_t_train_norm, trainlabels_arr)
            
            #### Results
            pred_train = clf.predict(X_t_train_norm)
            acc_final_tr = balanced_accuracy_score(trainlabels_arr,pred_train)
            acc_eval_tr.append(acc_final_tr)
            probs_train = clf.predict_proba(X_t_train_norm)
            lr_auc_tr = roc_auc_score(np.array(trainlabels_arr), np.array(probs_train[:, 1]))
            lr_auc_eval_tr.append(lr_auc_tr)
            perf_tr=perf_measure(pred_train,trainlabels_arr)

            
            #score = clf.score(X_t_test, testlabels_arr)
            pred_test = clf.predict(X_t_test_norm)
            acc_final_tt = balanced_accuracy_score(testlabels_arr,pred_test)
            acc_eval_tt.append(acc_final_tt)
            probs_test = clf.predict_proba(X_t_test_norm)
            lr_auc_tt = roc_auc_score(np.array(testlabels_arr), np.array(probs_test[:, 1]))
            lr_auc_eval_tt.append(lr_auc_tt)
            perf_tt=perf_measure(pred_test,testlabels_arr)

                      
        accs_media_tr[i]   = np.mean(np.array(acc_eval_tr))
        accs_media_tt[i]   = np.mean(np.array(acc_eval_tt))
        lr_auc_medio_tr[i] = np.mean(np.array(lr_auc_eval_tr))
        lr_auc_medio_tt[i] = np.mean(np.array(lr_auc_eval_tt))

      
    print('---------------- Accs FPs Kfold ----------------')
    print(f'>>> Acc media train: {np.mean(accs_media_tr)} +- {np.std(accs_media_tr)}')
    print(f'>>> Acc media test: {np.mean(accs_media_tt)} +- {np.std(accs_media_tt)}')
    print(f'>>> ROC AUC score medio train = {np.mean(lr_auc_medio_tr)} +- {np.std(lr_auc_medio_tr)} ')   
    print(f'>>> ROC AUC score medio test = {np.mean(lr_auc_medio_tt)} +- {np.std(lr_auc_medio_tt)}') 

    accs_media_tr = np.zeros([iterations,1])
    lr_auc_medio_tr = np.zeros([iterations,1])   


    print("----------- RESUSTITUCION ------------")    
    for i in range(iterations):  
        
        datalabels = datalabels[orden[:,i]]  
    
    
        if FES == 0:
            d=Npca
        elif FES == 1:
            d=Npls
        else:
            d=data.shape[1]
        alpha = 0.05
        #boundVC,_ = upper_bounds(data.shape[0],d,alpha)
        
          
        history_results = {}
        
        trainlabels_arr = datalabels
        clase1 = np.sum(trainlabels_arr==0)
        clase2 = np.sum(trainlabels_arr==1)
        nsamples = [clase1,clase2]
        # normedWeights = [1 - (x / sum(nsamples)) for x in nsamples]
        
        # class_weight = {0 : (trainlabels_arr == 1).sum() / (trainlabels_arr == 0).sum(), 
        #                 1 : 1.}
        
        data_train = np.copy(data)
        if len(data.shape) > 2:
            data_train = np.reshape(data_train,[data_train.shape[0],data_train.shape[1]*data_train.shape[2]])
      
        
        if FES == 0:
            pca = PCA(n_components=Npca)
            pca.fit(data_train)
            X_t_train = pca.transform(data_train)
        elif FES == 1:
            a=1 #pls
            plsca = PLSRegression(n_components=Npls)   
            plsca.fit(data_train, trainlabels_arr.flatten()) 
            X_t_train,_ = plsca.transform(data_train, trainlabels_arr.flatten())
        else: 
            X_t_train = np.copy(data_train)

            
        X_t_train_norm=scaler.fit_transform(X_t_train)   
               
        clf = SVC(kernel='linear',class_weight='balanced',probability=True)
        clf.fit(X_t_train_norm, trainlabels_arr)
        
        #### Results
        pred_train = clf.predict(X_t_train_norm)
        acc_tr = balanced_accuracy_score(trainlabels_arr,pred_train)
        boundVC = bound_bayes(data.shape[0],alpha,0.95,clf, acc_tr)
        acc_eval_tr = acc_tr - boundVC
        probs_train = clf.predict_proba(X_t_train_norm)
        lr_auc_eval_tr = roc_auc_score(np.array(trainlabels_arr), np.array(probs_train[:, 1]))
        
    
        accs_media_tr[i]   = acc_eval_tr
        lr_auc_medio_tr[i] = lr_auc_eval_tr   
    
    print('---------------- Accs FPs RUB ----------------')
    print(f'>>> Acc train: {np.mean(accs_media_tr)+boundVC} +- {np.std(accs_media_tr)}')
    print(f'>>> Bound RUB: {boundVC}')
    print(f'>>> Acc RUB train: {np.mean(accs_media_tr)} +- {np.std(accs_media_tr)}')
    print(f'>>> ROC AUC score medio train = {lr_auc_eval_tr}')   

    pvalueK = (np.sum((1-accs_media_tt)<=(1-0.7417))+1)/1001
    pvalueRUB = (np.sum((1-accs_media_tr)<=(1-0.8484))+1)/1001


