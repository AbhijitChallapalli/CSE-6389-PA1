import torch
import torch.nn as nn
import os

def visualize_predictions(model, dataloader, max_subjects=10, grid_cols=5):
    """
    Build 3 grid images (axial/coronal/sagittal), raw counts in titles.
    """
    model.eval()
    import math
    all_axials, all_coronals, all_sagittals = [], [], []
    preds, labels = [], []
    with torch.no_grad():
        for inputs, labs in dataloader:
            inputs, labs = inputs.to(device), labs.to(device)
            outputs = model(inputs)
            _, pr = torch.max(outputs, 1)
            for i in range(inputs.size(0)):
                vol = inputs[i].cpu().numpy()[0]  # [D,H,W]
                d,h,w = vol.shape
                all_axials.append(vol[d//2,:,:])
                all_coronals.append(vol[:,h//2,:])
                all_sagittals.append(vol[:,:,w//2])
                preds.append(int(pr[i].item()))
                labels.append(int(labs[i].item()))
                if len(preds) >= max_subjects:
                    break
            if len(preds) >= max_subjects:
                break

    def _save_grid(slices_2d, preds, labels, path, title, cols):
        import matplotlib.pyplot as plt
        n = len(slices_2d); rows = int(math.ceil(n/cols))
        plt.figure(figsize=(cols*2.2, rows*2.6))
        for i, sl in enumerate(slices_2d):
            ax = plt.subplot(rows, cols, i+1)
            ax.imshow(np.rot90(sl), cmap="gray")
            ax.set_title(f"Pred: {preds[i]}, Label: {labels[i]}")
            ax.axis("off")
        plt.suptitle(title); plt.tight_layout(rect=(0,0,1,0.97))
        plt.savefig(path, dpi=160); plt.close()

    os.makedirs("./", exist_ok=True)
    _save_grid(all_axials,    preds, labels, "./grid_axial.png",    "Axial Slices", grid_cols)
    _save_grid(all_coronals,  preds, labels, "./grid_coronal.png",  "Coronal Slices", grid_cols)
    _save_grid(all_sagittals, preds, labels, "./grid_sagittal.png", "Sagittal Slices", grid_cols)
