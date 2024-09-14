import numpy as np
import random
import torch
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score, f1_score, recall_score

from torch.utils.data import DataLoader, TensorDataset
from utils.MIMIC_utils import CustomDataset, load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def prepare_dataloader(delete,SEED=12345):
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    dataset = CustomDataset(time_range=(0, 6))

    if delete:
        # Ignore minority classes with fewer than 2000 samples
        dataset = ignore_minority_class(dataset, threshold=1000)

    return dataset.x, np.argmax(dataset.y, axis = 1)


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