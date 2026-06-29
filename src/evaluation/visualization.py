import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import umap
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Delay', 'Current', 'Past Due+'],
                yticklabels=['No Delay', 'Current', 'Past Due+'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

def plot_roc_curves(y_true, y_prob, save_path=None):
    plt.figure(figsize=(10, 8))
    
    # One-hot encode targets
    y_true_oh = np.zeros((len(y_true), 3))
    y_true_oh[np.arange(len(y_true)), y_true] = 1
    
    classes = ['No Delay', 'Current', 'Past Due+']
    colors = ['blue', 'orange', 'red']
    
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc:.2f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (OVR)')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved ROC curves to {save_path}")
    else:
        plt.show()

def plot_embeddings_umap(embeddings, labels, save_path=None, sample_size=5000):
    """Project 64D embeddings to 2D using UMAP."""
    if len(embeddings) > sample_size:
        # Sample for faster UMAP
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        emb_sample = embeddings[idx]
        lbl_sample = labels[idx]
    else:
        emb_sample = embeddings
        lbl_sample = labels
        
    logger.info("Computing UMAP projection...")
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(emb_sample)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                          c=lbl_sample, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=[0, 1, 2], format=plt.FuncFormatter(lambda val, loc: ['No Delay', 'Current', 'Past Due+'][int(val)]))
    plt.title('UMAP Projection of Customer Embeddings')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved UMAP plot to {save_path}")
    else:
        plt.show()
