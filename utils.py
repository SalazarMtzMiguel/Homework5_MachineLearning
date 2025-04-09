# utils.py
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

def compute_pca(data, n_components):
    mean = jnp.mean(data, axis=0)
    centered_data = data - mean
    cov_matrix = jnp.dot(centered_data.T, centered_data) / (centered_data.shape[0] - 1)
    eigvals, eigvecs = jnp.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = jnp.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    components = eigvecs[:, :n_components]
    
    # Project data
    transformed_data = jnp.dot(centered_data, components)
    
    return transformed_data, {'mean': mean, 'components': components}

def plot_roc_curve(y_true, y_pred, n_classes=2):
    plt.figure(figsize=(8, 6))
    if n_classes == 2:  # Caso binario
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    else:
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_precision_recall_curve(y_true, y_pred, n_classes=2):
    plt.figure(figsize=(8, 6))
    if n_classes == 2:  # Caso binario
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.2f})')
    else:
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'Class {i} (AUC = {pr_auc:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

def k_fold_cross_validation(x, y, k, model_fn, train_fn, epochs, batch_size, lr):
    fold_size = len(x) // k
    accuracies = []
    
    for i in range(k):
        # Split data
        val_start = i * fold_size
        val_end = (i + 1) * fold_size
        
        x_val = x[val_start:val_end]
        y_val = y[val_start:val_end]
        
        x_train = jnp.concatenate([x[:val_start], x[val_end:]], axis=0)
        y_train = jnp.concatenate([y[:val_start], y[val_end:]], axis=0)
        
        # Initialize and train model
        key = jax.random.PRNGKey(config['random_seed'])
        params = model_fn(key)
        
        for epoch in range(epochs):
            params, _ = train_fn(params, x_train, y_train, batch_size, lr)
        
        # Evaluate
        _, accuracy = evaluate(params, x_val, y_val)
        accuracies.append(accuracy)
    
    return jnp.mean(jnp.array(accuracies))