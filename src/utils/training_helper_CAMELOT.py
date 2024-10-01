import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_device(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tsaug
import time
from sklearn import cluster
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.special import softmax

from src.models.MultiTaskClassification import NonLinClassifier, MetaModel_AE
from src.models.model import CNNAE, DiffusionAE, AttenDiffusionAE, TimeAttentionCNNAE, TransformerAE
from src.utils.saver import Saver
from src.utils.utils import readable, reset_seed_, reset_model, flip_label, remove_empty_dirs, \
    evaluate_class, to_one_hot,small_loss_criterion_EPS, select_class_by_class, FocalLoss, CentroidLoss, reduce_loss, cluster_accuracy
from sklearn.cluster import KMeans

from src.plot.visualization import t_sne,t_sne_during_train
from models.camelot import CamelotModel


######################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = shutil.get_terminal_size().columns




def save_model_and_sel_dict(model,args,sel_dict=None):
    model_state_dict = model.state_dict()
    datestr = time.strftime(('%Y%m%d'))
    model_to_save_dir = os.path.join(args.basicpath, 'src', 'model_save', args.dataset)
    if not os.path.exists(model_to_save_dir):
        os.makedirs(model_to_save_dir, exist_ok=True)

    if args.label_noise == -1:
        label_noise = 'inst{}'.format(int(args.ni * 100))
    elif args.label_noise == 0:
        label_noise = 'sym{}'.format(int(args.ni * 100))
    else:
        label_noise = 'asym{}'.format(int(args.ni * 100))
    filename = os.path.join(model_to_save_dir, args.model)
    if sel_dict is not None:
        filename_sel_dict = '{}{}_{}_{}_sel_dict.npy'.format(filename, args.aug, label_noise, datestr)
        np.save(filename_sel_dict, sel_dict)  # save sel_ind
    filename = '{}{}_{}_{}.pt'.format(filename, args.aug, label_noise, datestr)
    torch.save(model_state_dict, filename)  # save model

def test_step(data_loader, model,model2=None):
    model = model.eval()
    if model2 is not None:
        model2 = model2.eval()

    yhat = []
    ytrue = []

    for x, y in data_loader:
        x = x.to(device)

        if model2 is not None:
            logits1 = model(x)
            logits2 = model2(x)
            logits = (logits1 + logits2) / 2
        else:
            logits = model(x)

        yhat.append(logits.detach().cpu().numpy())
        try:
            y = y.cpu().numpy()
        except:
            y = y.numpy()
        ytrue.append(y)

    yhat = np.concatenate(yhat,axis=0)
    ytrue = np.concatenate(ytrue,axis=0)
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)
    accuracy = accuracy_score(ytrue, y_hat_labels)
    f1_weighted = f1_score(ytrue, y_hat_labels, average='weighted')
    f1_macro= f1_score(ytrue, y_hat_labels, average = 'macro')
    
    
    
    # Compute Confusion Matrix
    cm = confusion_matrix(ytrue, y_hat_labels)
    print("Confusion Matrix:\n", cm) 
    
    # Classification Report
    class_report = classification_report(ytrue, y_hat_labels)
    print("Classification Report:\n", class_report) 
    
    return accuracy, f1_weighted


def train_eval_model(model, x_train, x_test, Y_train, Y_test, Y_train_clean,
                     ni, args, saver, plt_embedding=True, plt_cm=True):

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                  torch.from_numpy(np.arange(len(Y_train))), torch.from_numpy(Y_train_clean)) # 'Y_train_clean' is used for evaluation instead of training.

    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers)

    # compute noise prior
    ######################################################################################################
    # Train model

    model, test_results_last_ten_epochs = train_model(model, train_loader, test_loader, args,
                                          train_dataset=train_dataset,saver=saver)
    print('Train ended')

    ########################################## Eval ############################################

    # save test_results: test_acc(the final model), test_f1(the final model), avg_last_ten_test_acc, avg_last_ten_test_f1
    # test_results = evaluate_class(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
    #                               'Test', True, plt_cm=plt_cm, plt_lables=False) # evaluate_class will evaluate the final model.
    test_results = dict()
    test_results['acc'] = test_results_last_ten_epochs['last_ten_test_acc'][-1]
    test_results['f1_weighted'] = test_results_last_ten_epochs['last_ten_test_f1'][-1]
    test_results['avg_last_ten_test_acc'] = np.mean(test_results_last_ten_epochs['last_ten_test_acc'])
    test_results['avg_last_ten_test_f1'] = np.mean(test_results_last_ten_epochs['last_ten_test_f1'])

    #############################################################################################
    plt.close('all')
    torch.cuda.empty_cache()
    return test_results


def main_wrapper_CAMELOT(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=None):
    class SaverSlave(Saver):
        def __init__(self, path):
            super(Saver)
            self.args=args
            self.path = path
            self.makedir_()
            # self.make_log()

    classes = len(np.unique(Y_train_clean))
    args.nbins = classes

    ######################################################################################################
    print('Num Classes: ', classes)
    print('Train:', x_train.shape, Y_train_clean.shape, [(Y_train_clean == i).sum() for i in np.unique(Y_train_clean)])
    print('Test:', x_test.shape, Y_test_clean.shape, [(Y_test_clean == i).sum() for i in np.unique(Y_test_clean)])
    saver.append_str(['Train: {}'.format(x_train.shape),
                      'Test: {}'.format(x_test.shape), '\r\n'])

    ######################################################################################################
    # Main loop
    if seed is None:
        seed = np.random.choice(1000, 1, replace=False)
        
        
    model = CamelotModel(input_shape=(x_train.shape[1], x_train.shape[2]), seed=seed, output_dim=classes, num_clusters=10, latent_dim=64)
    
    nParams = sum([p.nelement() for p in model.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % ('CNN', readable(nParams))
    print(s)
    saver.append_str([s])

    print('#' * shutil.get_terminal_size().columns)
    print('RANDOM SEED:{}'.format(seed).center(columns))
    print('#' * shutil.get_terminal_size().columns)

    args.seed = seed

    ni = args.ni
    saver_slave = SaverSlave(os.path.join(saver.path, f'seed_{seed}', f'ratio_{ni}'))
    # True or false
    print('+' * shutil.get_terminal_size().columns)
    print('Label noise ratio: %.3f' % ni)
    print('+' * shutil.get_terminal_size().columns)

    reset_seed_(seed)
    model = reset_model(model)

    Y_train, mask_train = flip_label(x_train, Y_train_clean, ni, args)
    Y_test = Y_test_clean

    test_results = train_eval_model(model, x_train, x_test, Y_train,
                                                   Y_test, Y_train_clean,
                                                   ni, args, saver_slave,
                                                   plt_embedding=args.plt_embedding,
                                                   plt_cm=args.plt_cm)
    remove_empty_dirs(saver.path)

    return test_results



def plot_train_loss_and_test_acc(avg_train_losses,test_acc_list,args,pred_precision=None,saver=None,save=False,aug_accs=None):


    plt.gcf().set_facecolor(np.ones(3) * 240 / 255)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    l1 = ax.plot(avg_train_losses,'-', c='orangered', label='Training loss', linewidth=1)
    l2 = ax2.plot(test_acc_list, '-', c='blue', label='Test acc', linewidth=1)
    l3 = ax2.plot(pred_precision,'-',c='green',label='Sample_sel acc',linewidth=1)

    if len(aug_accs)>0:
        l4 = ax2.plot(aug_accs, '-', c='yellow', label='Aug acc', linewidth=1)
        lns = l1 + l2 + l3+l4
    else:
        lns = l1 + l2 + l3

    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs,loc='upper right')
    # plt.legend(handles=[l1,l2],labels=["Training loss","Test acc"],loc='upper right')

    plt.axvline(args.warmup,color='g',linestyle='--')

    ax.set_xlabel('epoch',  size=18)
    ax.set_ylabel('Train loss',size=18)
    ax2.set_ylabel('Test acc',  size=18)
    plt.gcf().autofmt_xdate()
    plt.title(f'Model:new model dataset:{args.dataset}')
    plt.grid(True)

    plt.tight_layout()

    saver.save_fig(fig, name=args.dataset)


def train_model(data_loader, model, optimizer, criterion, loss_all=None, epoch=0, args=None, sel_dict=None):
    
    global_step = 0
    aug_step = 0
    avg_accuracy = 0.
    avg_accuracy_aug = 0.
    avg_loss = 0.
    model = model.train()
    classes = args.nbins
    p = torch.ones(classes).to(device) / classes
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)



    model.initialize((data_loader.tensors[0],data_loader.tensors[1]))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cluster_optim = torch.optim.Adam([model.cluster_rep_set], lr=0.001)

    lr_scheduler = MyLRScheduler(
        optimizer, patience=15, min_lr=0.001, factor=0.25)
    cluster_lr_scheduler = MyLRScheduler(
        cluster_optim, patience=15, min_lr=0.001, factor=0.25)

    loss_mat = np.zeros((300, 4, 2))

    for i in trange(300):
        for step, (x_train, y_train) in enumerate(data_loader):
            optimizer.zero_grad()
            cluster_optim.zero_grad()
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            y_pred, probs = model.forward_pass(x_train)

            loss_weights = class_weight(y_train)

            common_loss = calc_pred_loss(y_train, y_pred, loss_weights)
            
            pred_loss = common_loss + \
                calc_l1_l2_loss(
                    layers=[model.Predictor.fc2, model.Predictor.fc3])
            pred_loss.backward(retain_graph=True, inputs=list(
                model.Predictor.parameters()))

            enc_loss = common_loss + model.alpha * calc_dist_loss(probs) + \
                + calc_l1_l2_loss(part=model.Encoder)
            enc_loss.backward(retain_graph=True, inputs=list(
                model.Encoder.parameters()))

            idnetifier_loss = common_loss + model.alpha * calc_dist_loss(probs) + \
                + calc_l1_l2_loss(layers=[model.Identifier.fc2])
            idnetifier_loss.backward(
                retain_graph=True, inputs=list(model.Identifier.parameters()))
                

            clus_loss = common_loss + model.beta * \
                calc_clus_loss(model.cluster_rep_set)
            clus_loss.backward(inputs=model.cluster_rep_set)

            optimizer.step()
            cluster_optim.step()

            loss_mat[i, 0, 0] += enc_loss.item()
            loss_mat[i, 1, 0] += idnetifier_loss.item()
            loss_mat[i, 2, 0] += pred_loss.item()
            loss_mat[i, 3, 0] += clus_loss.item()
            
        
        lr_scheduler.step(loss_mat[i, 0, 1])
        cluster_lr_scheduler.step(loss_mat[i, 0, 1])
        
        


    return (avg_accuracy / global_step, avg_accuracy_aug / aug_step), avg_loss / global_step, model