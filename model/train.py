from cnn_model import ECGCNN1D

import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train(X, y, groups, n_folds=10, n_epochs=50, lr=0.00003):


    """
    Function:
        Train a 1D CNN model on segmented EKG data and their associated binary labels, and extract metrics on model performance.

    Parameters: 
        X (arr): Epochs by heartbeat data.
        y (arr): Epoch labels.
        groups (arr): Grouped by record.

    Outputs:
        fold_results (list): List of best epoch results in each fold.
        history_result (list): List of fold's epoch's performance.
    """

    # Set a random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Convert arrays to PyTorch tensor data type
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Split data by fold while preserving class balance and keeping records separate
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    split = gkf.split(X.numpy(), y.squeeze(1).numpy(), groups=groups)

    fold_results = []
    history_results = []

    # Loop through each cross validation fold
    for fold, (train_idx, test_idx) in enumerate(split, start=1):
        
        # Create train and test splits
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"Fold {fold}")

        # Wrap tensors into PyTorch datasets
        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)

        y_train_flat = y_train.squeeze(1).long()
        n_pos = (y_train_flat == 1).sum().item()
        n_neg = (y_train_flat == 0).sum().item()

        class_weights = torch.tensor([1.0 / max(n_neg, 1), 1.0 / max(n_pos, 1)], dtype=torch.float32) # minority classes get a bigger weight
        sample_weights = class_weights[y_train_flat]

        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )

        # Only training set gets minority/majority class weighting
        train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        # 1D CNN
        model = ECGCNN1D()

        criterion = nn.BCEWithLogitsLoss()

        # Optimize model weights
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        fold_history = []
        best_fold_result = None
        best_auc = -np.inf

        # Train for the number of epochs specified 
        for epoch in range(n_epochs):

            model.train()

            running_train_loss = 0.0

            # loop through each batch
            for xb, yb in train_loader:

                optimizer.zero_grad() # clear previous gradients
                logits = model(xb) # compute predicted logits
                loss = criterion(logits, yb) # compute batch loss
                loss.backward() # compute gradients
                optimizer.step() # update model parameters

                running_train_loss += loss.item() * xb.size(0)
            
            # Average training loss across all train samples
            train_loss = running_train_loss / len(train_loader.dataset)

            model.eval()

            running_val_loss = 0.0

            all_probs = []
            all_preds = []
            all_true = []

            with torch.no_grad():
                for xb, yb in test_loader:

                    logits = model(xb)
                    loss = criterion(logits, yb) # compute validation loss
                    probs = torch.sigmoid(logits) # convert logits to probabilities
                    preds = (probs >= 0.5).float() # convert probabilities to binary predictions

                    running_val_loss += loss.item() * xb.size(0)

                    all_probs.append(probs)
                    all_preds.append(preds)
                    all_true.append(yb)
            
            val_loss = running_val_loss / len(test_loader.dataset)

            # Concatenate batch outputs into validation numpy arrays
            y_prob = torch.cat(all_probs).numpy().ravel()
            y_pred = torch.cat(all_preds).numpy().ravel()
            y_true = torch.cat(all_true).numpy().ravel()

            # Metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc = roc_auc_score(y_true, y_prob)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            epoch_result = {
                'epoch': epoch+1,
                'train_loss': train_loss, 
                'val_loss': val_loss,
                'accuracy': acc, 
                'precision': prec, 
                'recall': rec, 
                'F1': f1, 
                'AUC': auc
            }
            fold_history.append(epoch_result)

            print(
                f"Epoch {epoch+1:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Acc: {acc:.4f} | "
                f"Prec: {prec:.4f} | "
                f"Rec: {rec:.4f} | "
                f"F1: {f1:.4f} | "
                f"AUC: {auc:.4f}"
            )

            # Selecting epoch results that had highest auc in a fold
            score_for_selection = auc if not np.isnan(auc) else -np.inf
            if score_for_selection > best_auc:
                best_auc = score_for_selection
                best_fold_result = {
                    "fold": fold,
                    "best_epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "F1": f1,
                    "auc": auc,
                    "cm": cm,
                } 

        fold_results.append(best_fold_result)
        history_results.append({
            "fold": fold,
            "history": fold_history
        })

    return fold_results, history_results