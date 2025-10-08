import os, math
import numpy as np
import matplotlib.pyplot as plt

def _center_index(n: int) -> int:
    return max(0, n // 2)

def _resolve_index(n: int, idx):
    if isinstance(idx, str) and idx.lower() == "center":
        return _center_index(n)
    try:
        i = int(idx)
        return max(0, min(n - 1, i))
    except Exception:
        return _center_index(n)

def _imshow(ax, img, cmap="gray"):
    ax.imshow(np.rot90(img), cmap=cmap)
    ax.axis("off")

def save_three_orthogonal_slices(volume: np.ndarray,
                                 out_path: str,
                                 title: str = None,
                                 cmap: str = "gray",
                                 idx="center") -> None:
    assert volume.ndim == 3
    d, h, w = volume.shape
    i = _resolve_index(d, idx)
    j = _resolve_index(h, idx)
    k = _resolve_index(w, idx)
    axial, coronal, sagittal = volume[i, :, :], volume[:, j, :], volume[:, :, k]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    _imshow(axes[0], axial, cmap); _imshow(axes[1], coronal, cmap); _imshow(axes[2], sagittal, cmap)
    if title: fig.suptitle(title, y=0.98, fontsize=10)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.02, hspace=0.02)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=170); plt.close(fig)

def save_slice_grid(slices_2d, preds, labels, out_path: str,
                    cols: int = 5, cmap: str = "gray", title: str = None) -> None:
    assert len(slices_2d) == len(preds) == len(labels)
    n = len(slices_2d)
    cols = max(1, int(cols))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.1, rows*2.1))
    axes = np.array(axes).reshape(rows, cols)

    k = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if k < n:
                _imshow(ax, slices_2d[k], cmap)
                ax.text(0.02, 0.06, f"Pred:{preds[k]}  Lab:{labels[k]}",
                        color="w", fontsize=8, transform=ax.transAxes,
                        bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"))
            else:
                ax.axis("off")
            k += 1

    if title: fig.suptitle(title, y=0.99, fontsize=12)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01, wspace=0.02, hspace=0.02)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=170); plt.close(fig)
