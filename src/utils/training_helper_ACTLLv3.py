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


######################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = shutil.get_terminal_size().columns





def temperature(x, th_low, th_high, low_val, high_val):
    if x < th_low:
        return low_val
    elif th_low <= x < th_high:
        return (x - th_low) / (th_high - th_low) * (high_val - low_val) + low_val
    else:  # x == th_high
        return high_val
    
    
######################################################################################################
class CentroidLoss(nn.Module):
    """
    Centroid loss - Constraint Clustering loss of SREA
    """

    def __init__(self, feat_dim, num_classes, reduction='mean'):
        super(CentroidLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        self.reduction = reduction
        self.rho = 1.0

    def forward(self, h, y):
        C = self.centers
        norm_squared = torch.sum((h.unsqueeze(1) - C) ** 2, 2)
        # Attractive
        distance = norm_squared.gather(1, y.unsqueeze(1)).squeeze(-1)
        # Repulsive
        logsum = torch.logsumexp(-torch.sqrt(norm_squared), dim=1)
        loss = reduce_loss(distance + logsum, reduction=self.reduction)
        # Regularization
        reg = self.regularization(reduction='sum')
        return loss + self.rho * reg

    def regularization(self, reduction='sum'):
        C = self.centers
        pairwise_dist = torch.cdist(C, C, p=2) ** 2
        pairwise_dist = pairwise_dist.masked_fill(
            torch.zeros((C.size(0), C.size(0))).fill_diagonal_(1).bool().to(device), float('inf'))
        distance_reg = reduce_loss(-(torch.min(torch.log(pairwise_dist), dim=-1)[0]), reduction=reduction)
        return distance_reg
    






def add_to_confident_set_id(args=None,confident_set_id=None,train_dataset=None,epoch=None,conf_num=None):

    xs, ys, _, y_clean = train_dataset.tensors
    ys=ys.cpu().numpy()
    y_clean=y_clean.cpu().numpy()
    TP_all=0
    FP_all=0
    for i in range(args.nbins):
        confnum_row = dict()
        confnum_row['epoch'] = epoch
        if args.sel_method == 5:
            confnum_row['method']='BMM'
        elif args.sel_method == 2:
            confnum_row['method'] = 'GMM'
        else: # sel_method in [1,2]
            confnum_row['method'] = 'Small Loss'
        confnum_row['label']=i
        confnum_row['total']=sum(ys[confident_set_id]==i)
        confnum_row['TP'] = sum((y_clean[confident_set_id][ys[confident_set_id]==i]==i))
        TP_all=TP_all+confnum_row['TP']
        confnum_row['FP'] = sum((y_clean[confident_set_id][ys[confident_set_id]==i]!=i))
        FP_all =FP_all+ confnum_row['FP']
        confnum_row['seed'] = args.seed

        conf_num.append(confnum_row)
    if (TP_all + FP_all) > 0:
        estimate_noise_rate = TP_all / (TP_all + FP_all)
    else:
        estimate_noise_rate = 0  # or handle it in a way that suits your use case
    return conf_num, estimate_noise_rate








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


def train_model(model, train_loader, test_loader, args,train_dataset=None,saver=None):
    
    
    if args.modelloss == 'Focal':
        criterion = FocalLoss(gamma=2.0, reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduce=False)
    
    classes = args.nbins
    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=args.lr, weight_decay=args.l2penalty, eps=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    
    # learning history
    train_acc_list = []
    train_acc_list_aug = []
    train_avg_loss_list = []
    test_acc_list = []
    test_f1s = []
    
    
    p = torch.ones(classes).to(device) / classes
    history_track = args.track
    yhat_hist = torch.zeros(train_loader.dataset.tensors[0].size(0), classes, history_track).to(device)
    
    
    try:
        loss_all = np.zeros((args.num_training_samples, args.epochs))
        conf_num = []
        hard_num = []
        less_conf_num = []
        
        for e in range(args.epochs):
            sel_dict = {'sel_ind': [], 'lam': [], 'mix_ind': []}
            # training step
            if e <= args.warmup:
                
                train_accuracy, avg_loss, model_new, y_hat_hist = warmup_ACTLL(data_loader=train_loader,model=model, 
                                                                                              
                                                                            yhat_hist=yhat_hist, 
                                                                            optimizer=optimizer,
                                                                            criterion=criterion,
                                                                            epoch=e,
                                                                            loss_all=loss_all,
                                                                            args=args)
                
            else:
                train_accuracy, avg_loss, model_new, confident_set_id, hard_set_id, less_confident_set_id, y_corr_all = train_step_ACTLLv3(
                                                                            data_loader=train_loader,
                                                                            model=model,                    
                                                                            optimizer=optimizer,
                                                                            loss_all=loss_all,
                                                                            criterion=criterion,
                                                                            yhat_hist = yhat_hist,
                                                                            epoch=e,
                                                                            args=args, sel_dict=sel_dict)

                if args.confcsv is not None: # save confident samples' id to visualize
                    conf_num, _ = add_to_confident_set_id(args=args,
                                                       confident_set_id=confident_set_id.astype(int),
                                                       train_dataset=train_dataset, epoch=e,
                                                       conf_num=conf_num)
                    
                    hard_num, _ = add_to_confident_set_id(args=args,
                                                       confident_set_id=hard_set_id.astype(int),
                                                       train_dataset=train_dataset, epoch=e,
                                                       conf_num=hard_num)
                    
                    less_conf_num, _ = add_to_confident_set_id(args=args,
                                                       confident_set_id=less_confident_set_id.astype(int),
                                                       train_dataset=train_dataset, epoch=e,
                                                       conf_num=less_conf_num)
                
                
            model = model_new

            if args.tsne_during_train and args.seed == args.manual_seeds[0] and e in args.tsne_epochs:
                xs, ys, _, y_clean = train_dataset.tensors
                with torch.no_grad():
                    t_sne_during_train(xs, ys, y_clean, model=model, tsne=True, args=args,sel_dict=sel_dict,epoch=e)

            # testing
            test_accuracy, f1 = test_step(data_loader=test_loader,
                                      model=model)

            # train results each epoch
            train_acc_list.append(train_accuracy[0])
            train_acc_list_aug.append(train_accuracy[1])
            train_acc_oir =train_accuracy[0]
            train_avg_loss_list.append(avg_loss)

            # test results each epoch
            test_acc_list.append(test_accuracy)
            test_f1s.append(f1)

            print(
                '{} epoch - Train Loss {:.4f}\tTrain accuracy {:.4f}\tTest accuracy {:.4f}'.format(
                    e + 1,
                    avg_loss,
                    train_acc_oir,
                    test_accuracy))
            scheduler.step()


    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')


    if args.confcsv is not None:
        csvpath = os.path.join(args.basicpath, 'src', 'bar_info')
        if not os.path.exists(csvpath):
            os.makedirs(csvpath)
        pd.DataFrame(conf_num).to_csv(os.path.join(csvpath, args.dataset + str(args.sel_method) + args.confcsv),
                                      mode='a', header=True)
        
    if args.hardcsv is not None:
        csvpath = os.path.join(args.basicpath, 'src', 'hard_info')
        if not os.path.exists(csvpath):
            os.makedirs(csvpath)
        pd.DataFrame(hard_num).to_csv(os.path.join(csvpath, args.dataset + str(args.sel_method) + args.hardcsv),
                                      mode='a', header=True)
        
    if args.lessconfcsv is not None:
        csvpath = os.path.join(args.basicpath, 'src', 'lessconf_info')
        if not os.path.exists(csvpath):
            os.makedirs(csvpath)
        pd.DataFrame(less_conf_num).to_csv(os.path.join(csvpath, args.dataset + str(args.sel_method) + args.lessconfcsv),
                                      mode='a', header=True)
        
        
    if args.save_model:
        save_model_and_sel_dict()
    if args.plt_loss_hist:
        plot_train_loss_and_test_acc(train_avg_loss_list,test_acc_list,args,pred_precision=train_acc_list,aug_accs=train_acc_list_aug,
                                     saver=saver,save=True)
    if args.plot_tsne and args.seed==args.manual_seeds[0]:
        xs,ys,_,y_clean = train_dataset.tensors
        datestr = time.strftime(('%Y%m%d'))
        with torch.no_grad():
            t_sne(xs, ys, y_clean,model=model, tsne=True, args=args,datestr=datestr,sel_dict=sel_dict)

    test_results_last_ten_epochs = dict()
    test_results_last_ten_epochs['last_ten_test_acc'] = test_acc_list[-10:]
    test_results_last_ten_epochs['last_ten_test_f1'] = test_f1s[-10:]
    return model, test_results_last_ten_epochs


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


def main_wrapper_ACTLLv3(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=None):
    class SaverSlave(Saver):
        def __init__(self, path):
            super(Saver)
            self.args=args
            self.path = path
            self.makedir_()
            # self.make_log()

    classes = len(np.unique(Y_train_clean))
    args.nbins = classes

    # Network definition
    classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                   norm=args.normalization)
    
    if args.AEChoice ==  'CNN':
        model = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                   seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                   padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)
    
    elif args.AEChoice == 'Diffusion':
        model = DiffusionAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                    seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                    padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)

    elif args.AEChoice == 'AttentionDiffusion':
        model = AttenDiffusionAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                    seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                    padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)
        
    elif args.AEChoice == 'TimeAtteCNN':
        
        # Define the model using the TimeAttentionCNNAE class
        model = TimeAttentionCNNAE(input_size=x_train.shape[2],               # Number of input features
                                num_filters=args.filters,                  # Number of filters for the CNN layers
                                embedding_dim=args.embedding_size,         # Embedding size for the final encoding
                                seq_len=x_train.shape[1],                  # Sequence length (time steps)
                                kernel_size=args.kernel_size,              # Kernel size for CNN layers
                                stride=args.stride,                        # Stride for CNN layers
                                padding=args.padding,                      # Padding for CNN layers
                                dropout=args.dropout,                      # Dropout rate
                                normalization=args.normalization,          # Normalization type (e.g., batch norm)
                                num_heads=4                                # Number of attention heads
                                ).to(device)
        
    elif args.AEChoice == 'Transformer':
        # Initialize the model for TransformerAE
        model = TransformerAE(
            input_size=x_train.shape[2],        # Input feature size (e.g., 9, like channels in CNN)
            embedding_dim=args.embedding_size,  # Latent space dimension (embedding size)
            num_heads=4,                        # Number of attention heads in the transformer encoder
            num_filters=args.filters,           # Number of filters for ConvDecoder
            seq_len=x_train.shape[1],           # Input sequence length (e.g., 6)
            kernel_size=args.kernel_size,       # Kernel size for ConvDecoder
            stride=args.stride,                 # Stride for ConvDecoder
            padding=args.padding,               # Padding for ConvDecoder
            dropout=args.dropout,               # Dropout rate for regularization
            dim_feedforward=4 * args.embedding_size  # Feedforward network dimension in Transformer
        ).to(device)
        
        
    
    else: # If no choice then CNN
        model = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                   seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                   padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = MetaModel_AE(ae=model, classifier=classifier, name='CNN').to(device)

    nParams = sum([p.nelement() for p in model.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % ('CNN', readable(nParams))
    print(s)
    saver.append_str([s])

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




def warmup_ACTLL(data_loader, model, yhat_hist, optimizer, criterion,epoch=None,
                         loss_all=None,args=None):
    
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.
    model = model.train()
    
    

    for batch_idx,(x, y_hat,x_idx,_) in enumerate(data_loader):
        if x.shape[0]==1:
            continue
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)

        if hasattr(model,'decoder'):
            h=model.encoder(x)
            hd=model.decoder(h)
            out=model.classifier(h.squeeze(-1))
            model_loss = criterion(out, y_hat)
            
            
            loss_all[x_idx, epoch] = model_loss.data.detach().clone().cpu().numpy()
            
            # ADD CLUSTERING MODULE LOSS

            model_loss = model_loss.sum() + nn.MSELoss(reduction='mean')(hd, x)
            # model_loss = model_loss.sum() + nn.MSELoss(reduction='mean')(hd, x) + loss_centroids(h.squeeze(-1), torch.argmax(out, dim=1)).mean()

        else:
            out = model(x)
            model_loss = criterion(out, y_hat).sum()

        ############################################################################################################


        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss = avg_loss + model_loss.item()

        # Compute accuracy
        acc = torch.eq(torch.argmax(out, 1), y_hat).float()
        avg_accuracy += acc.sum().cpu().numpy()
        global_step += len(y_hat)
        
        # Get historical prediction 
        # Append predictions
        
        # Accuracy on noisy labels
        prob = F.softmax(out, dim=1)
        

        
        yhat_hist[x_idx] = yhat_hist[x_idx].roll(1, dims=-1)  # Roll the elements along the last dimension
        yhat_hist[x_idx, :, 0] = prob.detach()  # Assign the probability to the first position

    
    return (avg_accuracy / global_step,0.), avg_loss / global_step, model, yhat_hist




def fft_augmentation(x, freq_scale=0.1):
    # Apply Fast Fourier Transform (FFT)
    fft_data = np.fft.fft(x.cpu().numpy(), axis=-1)
    
    # Modify low frequencies
    fft_data[:, :int(fft_data.shape[-1] * freq_scale)] = 0.0
    
    # Apply inverse FFT
    augmented_data = np.fft.ifft(fft_data, axis=-1)
    
    return torch.from_numpy(augmented_data.real).float().to(device)






########### CORRECT THE LESS CONFIDENT LABEL USING RESULT OF CLUSTERING OUTCOME###############
def label_correction(embedding, centers, y_obs, yhat_hist, w_yhat, w_c, w_obs, classes):
    
    
    # yhat from previous metwork prediction. - Network Ensemble
    steps = yhat_hist.size(-1)
    decay = torch.arange(0, steps, 1).float().to(device)
    decay = torch.exp(-decay / 2)
    yhat_hist = yhat_hist * decay
    yhat = yhat_hist.mean(dim=-1) * w_yhat
    

    # Label from clustering
    distance_centers = torch.cdist(embedding.squeeze(-1), centers)
    yc = F.softmin(distance_centers, dim=1).detach() * w_c

    # Observed - given - label (noisy)
    yobs = F.one_hot(y_obs, num_classes=classes).float() * w_obs
    
    
    # Label combining
    ystar = (yhat + yc + yobs) / 3
    ystar = torch.argmax(ystar, dim=1)
    
    
    return ystar



def hard_set_loss(hard_set_probs, y_hat, y_pred, out):
    """
    Compute the hard set loss using a weighted combination of noisy labels and model predictions.
    
    Args:
    - hard_set_probs (torch.Tensor): Confidence scores (weights) for each hard sample (w_i).
    - y_hat (torch.Tensor): Noisy labels (y_i) for the hard samples.
    - y_pred (torch.Tensor): Model's predicted labels (z_i) for the hard samples.
    - out (torch.Tensor): Model's output logits or probabilities (h_i).
    
    Returns:
    - loss (torch.Tensor): Computed loss for the hard set.
    """
    
    # Convert model logits to probabilities using softmax
    
    hard_set_probs = hard_set_probs.to(device)
    y_hat = y_hat.to(device)
    y_pred = y_pred.to(device)
    out = out.to(device)
    
    # Convert logits to log probabilities
    log_h = F.log_softmax(out, dim=1)  # log(h_i)

    # Create one-hot encodings for the labels
    y_one_hot = F.one_hot(y_hat, num_classes=out.size(1)).float()  # y_i
    z_one_hot = F.one_hot(y_pred, num_classes=out.size(1)).float()  # z_i
    
    
    # Compute the weighted combination of noisy labels and predicted labels
    # weighted_labels = (1 - hard_set_probs)[:, None] * y_one_hot + hard_set_probs[:, None] * z_one_hot
    
    weighted_labels = (1 - hard_set_probs)[:, None] * z_one_hot + hard_set_probs[:, None] * y_one_hot
    # Compute the loss
    loss = -torch.sum(weighted_labels * log_h, dim=1).mean()
    

    return loss, weighted_labels




def temperature_scaled_coefficients(epoch, start_epoch=200, total_epochs=300, T_corr=10.0, T_hard=10.0, k=0.1, max_corr=1.0, max_hard=1.0):
    """
    Temperature scaling for L_corr and L_hard coefficients.
    Coefficients increase gradually from start_epoch to total_epochs.
    :param epoch: Current epoch
    :param start_epoch: When to start increasing L_corr and L_hard
    :param total_epochs: Maximum number of epochs
    :param T_corr: Initial temperature for L_corr scaling
    :param T_hard: Initial temperature for L_hard scaling
    :param k: Steepness of the curve
    :param max_corr: Maximum value for L_corr coefficient
    :param max_hard: Maximum value for L_hard coefficient
    :return: Scaled coefficients for L_corr and L_hard
    """
    
    # Only start increasing the coefficients after the start_epoch
    if epoch < start_epoch:
        return 0.0, 0.0
    
    
    # Adjust temperature scaling based on the current epoch relative to the range from start_epoch to total_epochs
    epoch_in_range = epoch - start_epoch
    total_epochs_in_range = total_epochs - start_epoch
    
    # Exponential decay for T_corr and T_hard starting at start_epoch
    lambda_corr = max_corr / (1 + np.exp(-k * (epoch_in_range - T_corr)))
    lambda_hard = max_hard / (1 + np.exp(-k * (epoch_in_range - T_hard)))
    
    return lambda_corr, lambda_hard







def train_step_ACTLLv3(data_loader, model,  optimizer, criterion, yhat_hist, loss_all=None, epoch=0, args=None, sel_dict=None):
    
    global_step = 0
    aug_step = 0
    avg_accuracy = 0.
    avg_accuracy_aug = 0.
    avg_loss = 0.
    model = model.train()
    confident_set_id = np.array([])
    hard_set_id = np.array([])
    less_confident_set_id = np.array([])
    classes = args.nbins
    p = torch.ones(classes).to(device) / classes
    y_corr_all = torch.tensor(np.array([])).to(device)
    
    
    
    
    for batch_idx, (x, y_hat, x_idx, _) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)
        y = y_hat

        h = model.encoder(x)
        out = model.classifier(h.squeeze(-1))

        hd = model.decoder(h)
        recon_loss = nn.MSELoss(reduction='mean')(hd, x)
        
        # Accuracy on noisy labels
        prob = F.softmax(out, dim=1)
        prob_avg = torch.mean(prob, dim=0)

        loss = criterion(out, y_hat)
        
        y_pred = torch.argmax(prob, dim=1).to(device)

        if loss_all is not None:
            loss_all[x_idx, epoch] = loss.data.detach().clone().cpu().numpy()


        ################################# L_CONF ######################
        if args.sel_method == 5: # BMM
            L_conf, model_sel_idx, hard_set_idxs, less_confident_idxs, hard_set_probs = select_class_by_class(model_loss = loss, loss_all=loss_all, labels=y_hat, p_threshold=args.p_threshold,
                                                                args=args,
                                                                epoch=epoch, x_idxs=x_idx)
        elif args.sel_method == 2: # GMM
            L_conf, model_sel_idx, hard_set_idxs, less_confident_idxs,hard_set_probs = select_class_by_class(model_loss = loss, loss_all=loss_all, labels=y_hat, p_threshold=args.p_threshold,
                                                                args=args,
                                                                epoch=epoch, x_idxs=x_idx)
        else:
            L_conf, model_sel_idx, less_confident_idxs = small_loss_criterion_EPS(
                model_loss=loss,
                loss_all=loss_all,
                args=args,
                epoch=epoch,
                x_idxs=x_idx,
                labels=y_hat
            )
            # Convert all indices to sets for easier set operations
            all_indices = set(range(len(loss)))
            confident_set = set(model_sel_idx.tolist())  # Indices of confident samples
            less_confident_set = set(less_confident_idxs.tolist())  # Indices of less confident samples

            # Hard set includes indices that are not in either the confident or less confident set
            hard_set_indices = all_indices - confident_set - less_confident_set

            # Convert hard_set_indices back to a tensor
            hard_set_idxs = torch.tensor(list(hard_set_indices)).long()
            
            hard_set_probs = hard_set_probs = torch.tensor([]).long()
        
        ################################# L_aug #####################################
        if (batch_idx % args.arg_interval == 0) and len(model_sel_idx) > 0 and args.augment:
            x_aug = torch.from_numpy(
                tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                    x[model_sel_idx].cpu().numpy())).float().to(device)
            
            aug_step += 1
            if len(x_aug) == 1:  # Avoid bugs
                aug_model_loss = torch.tensor(0.)
                avg_accuracy_aug = 0.
            else:
                aug_h = model.encoder(x_aug)
                outx_aug = model.classifier(aug_h.squeeze(-1))
                y_hat_aug = y_hat[model_sel_idx]
                aug_model_loss = criterion(outx_aug, y_hat[model_sel_idx]).sum()
                avg_accuracy_aug += torch.eq(torch.argmax(outx_aug, 1), y_hat_aug).float().sum().cpu().numpy()

            if epoch == args.epochs - 1 or epoch in args.tsne_epochs:
                sel_dict['sel_ind'].append(x_idx[model_sel_idx].cpu().numpy())

        else:
            aug_model_loss = torch.tensor(0.)
            aug_step = 1 # prevent zero division
            avg_accuracy_aug = 0.
        
        
        ####################### L_HARD + DATA CORRECTION FOR HARD EXAMPLES ##############
        if len(hard_set_idxs) > 0 and args.hard:
            # Compute L_hard using a weighted combination of noisy labels and model's predicted labels
            L_hard, y_corr = hard_set_loss(hard_set_probs, y_hat[hard_set_idxs], y_pred[hard_set_idxs], out[hard_set_idxs])
            
            y_corr_label = torch.argmax(y_corr, dim=-1).view(-1)
            y_corr_all = torch.cat((y_corr_all, y_corr_label),dim = 0)

        else:
            L_hard = torch.tensor(0)
            
        
        ################## L_corr: Using Model Prediction Directly
        if len(less_confident_idxs) > 0 and args.corr:
            soft_targets = prob[less_confident_idxs].detach()  # Detach soft targets to prevent gradient backprop
            # Assuming y_hat contains class indices, convert it to one-hot vectors with the same number of classes as soft_targets
            num_classes = soft_targets.size(1)  # Assuming soft_targets has the correct number of classes

            # One-hot encode y_hat[less_confident_idxs] to match the dimensionality of soft_targets
            y_hat_one_hot = F.one_hot(y_hat[less_confident_idxs], num_classes=num_classes).float()

            # Now apply the KL divergence with the same shapes
            L_corr = F.kl_div(F.log_softmax(y_hat_one_hot, dim=-1), soft_targets, reduction='batchmean').mean()
            
        else:
            L_corr = torch.tensor(0)
            
            
    
        # Clamping to avoid extreme values
        recon_loss = torch.clamp(recon_loss, min=1e-8, max=1e8)
        # L_p = -torch.sum(torch.log(prob_avg) * p)  # Distribution regularization
        
        # args.L_p_coef * L_p
        
        
        
        lambda_corr, lambda_hard = temperature_scaled_coefficients(epoch, start_epoch=args.correct_start, total_epochs=args.epochs)
        
        model_loss =  L_conf + args.L_rec_coef * recon_loss \
                    + lambda_hard * L_hard + lambda_corr * L_corr \
                    + args.L_aug_coef * aug_model_loss
        
        
        # print(f"Batch {batch_idx + 1} | L_conf: {L_conf.item():.4f} | L_aug: {aug_model_loss.item():.4f} | L_rec: {recon_loss.item():.4f}")
        # print(f"L_corr: {L_corr.item():.4f} | L_hard: {L_hard.item():.4f} | L_p: {L_p.item():.4f}")

        # print(f"model_loss {model_loss.item():.4f}")
        
        # Loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    
        avg_loss += model_loss.item()

        # Compute accuracy
        acc1 = torch.eq(torch.argmax(out, 1), y).float()
        avg_accuracy += acc1.sum().cpu().numpy()

        global_step += 1

        confident_set_id = np.concatenate((confident_set_id, x_idx[model_sel_idx].cpu().numpy()))
        hard_set_id = np.concatenate((hard_set_idxs, x_idx[model_sel_idx].cpu().numpy()))
        less_confident_set_id = np.concatenate((less_confident_idxs, x_idx[model_sel_idx].cpu().numpy()))
        
        

        yhat_hist[x_idx] = yhat_hist[x_idx].roll(1, dims=-1)  # Roll the elements along the last dimension
        yhat_hist[x_idx, :, 0] = prob.detach()  # Assign the probability to the first position
        
        
        

    return (avg_accuracy / global_step, avg_accuracy_aug / aug_step), avg_loss / global_step, model, confident_set_id, hard_set_id, less_confident_set_id, y_corr_all