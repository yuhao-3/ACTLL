import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, classification_report

from torch.utils.data import DataLoader, TensorDataset
from utils.data_utils import CustomDataset, load_data
from utils.train_utils import MyLRScheduler, class_weight, calc_pred_loss, calc_clus_loss, calc_dist_loss, calc_l1_l2_loss
from utils.model_utils import CNNModel
import torch.nn.functional as F




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_dataloader_synthetic(noise_level_str, SEED=12345):
    # Generate synthetic data
    directory = f'../data/synthetic_camelot/noise_{noise_level_str}'
    
    X = np.load(f'{directory}/X.npy')
    y_clean = np.load(f'{directory}/y.npy')
    y_noisy = np.load(f'{directory}/y_noisy_{noise_level_str}.npy')

    
    # ######## CLEAN #######
    # # Split the dataset into training and validation sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y_clean, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    ###### NOISY ######
    
    # Split the dataset into training+validation and testing sets with clean labels for the test set
    X_train_val, X_test, y_train_val_clean, y_test = train_test_split(X, y_clean, test_size=0.2, random_state=42)

    # Extract noisy labels corresponding to X_train_val indices from y_noisy
    indices_train_val = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)[0]
    y_train_val_noisy = y_noisy[indices_train_val]

    # Split the training+validation set further, using noisy labels for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val_noisy, test_size=0.2, random_state=42)
    
    
    # Initialize model components
    
    x_train_tensor = torch.tensor(X_train, dtype=torch.float)
    x_val_tensor = torch.tensor(X_val, dtype = torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)
    y_val_tensor = torch.tensor(y_val, dtype = torch.float)
    x_test_tensor = torch.tensor(X_test, dtype = torch.float)
    y_test_tensor = torch.tensor(y_test, dtype = torch.float)
     
    # Create TensorDatasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create DataLoaders
    batch_size = 64  # You can adjust the batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader




# Assuming CustomDataset and load_data are defined elsewhere
def ignore_minority_class(dataset, threshold=1000):
    # Get labels and their counts (convert one-hot to class index)
    labels = np.argmax(dataset.y, axis=-1)
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # Identify classes with sufficient samples
    majority_classes = [cls for cls, count in class_counts.items() if count >= threshold]
    
    # Filter the dataset
    indices = [i for i, label in enumerate(labels) if label in majority_classes]
    filtered_dataset = dataset.get_subset(indices)
    
    
    # Relabel majority class samples to new one-hot encoding
    new_labels = {old_label: new_label for new_label, old_label in enumerate(majority_classes)}
    new_y = np.array([new_labels[label] for label in np.argmax(filtered_dataset.y, axis=-1)])
    new_y_one_hot = np.eye(len(majority_classes))[new_y]
    filtered_dataset.y = new_y_one_hot 

    return filtered_dataset



def prepare_dataloader(SEED=12345, delete=True):
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    dataset = CustomDataset(time_range=(0, 6))
    
    if delete:
        # Ignore minority classes with fewer than 2000 samples
        dataset = ignore_minority_class(dataset, threshold=1000)

    # Stratified Sampling for train and val
    train_idx, test_idx = train_test_split(np.arange(len(dataset)),
                                           test_size=0.4,
                                           random_state=SEED,
                                           shuffle=True,
                                           stratify=np.argmax(dataset.y, axis=-1))

    # Subset dataset for train and val
    train_val_dataset = dataset.get_subset(train_idx)
    test_dataset = dataset.get_subset(test_idx)
    

    train_idx, val_idx = train_test_split(np.arange(len(train_val_dataset)),
                                          test_size=0.4,
                                          random_state=SEED,
                                          shuffle=True,
                                          stratify=np.argmax(train_val_dataset.y, axis=-1))

    train_dataset = train_val_dataset.get_subset(train_idx)
    val_dataset = train_val_dataset.get_subset(val_idx)
    
    

    train_loader, val_loader, test_loader = load_data(
        train_dataset, val_dataset, test_dataset)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def get_test_results(model, test_loader):
    real, preds = [], []
    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model.forward_pass(x)
            preds.extend(list(y_pred.cpu().detach().numpy()))
            real.extend(list(y.cpu().detach().numpy()))
    return real, preds


def calc_metrics(real, preds):
    
    auc = roc_auc_score(real, preds, average=None)

    labels_true, labels_pred = np.argmax(
        real, axis=1), np.argmax(preds, axis=1)

    # Compute Macro F1
    f1 = f1_score(labels_true, labels_pred, average=None)
    # f1 = f1_score(labels_true, labels_pred, average='weighted')
    # Compute WEighted F1
    

    # Compute Recall
    rec = recall_score(labels_true, labels_pred, average=None)

    # Compute NMI
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    
    # Compute Confusion Matrix
    cm = confusion_matrix(labels_true, labels_pred)
    
    # Classification Report
    class_report = classification_report(labels_true, labels_pred)
    print("Classification Report:\n", class_report) 

    return auc, f1, rec, nmi, cm



