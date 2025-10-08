# src/vis/plots.py
import numpy as np
import matplotlib.pyplot as plt

def plot_train_loss(train_losses, out_path: str):
    plt.figure()
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker='o', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_confusion_matrix(cm, out_path: str, class_names=('health', 'patient')):
    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, interpolation='nearest', cmap='Reds')  # RAW COUNTS in Reds
    plt.title('Confusion Matrix'); plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
