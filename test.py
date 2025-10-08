import os, argparse, yaml, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MRIVolumeDataset, load_canonical, normalize_clip_z, resize_3d
from src.models.cnn3d import CNN3D
from src.utils.seed import set_all_seeds
from src.utils.metrics import compute_basic_metrics
from src.vis.plots import plot_confusion_matrix
from src.vis.slices import save_three_orthogonal_slices, save_slice_grid


def _load_auto_threshold(out_dir, default=0.5):
    p = os.path.join(out_dir, "threshold.txt")
    try:
        with open(p, "r") as f:
            return float(f.readline().strip())
    except Exception:
        return float(default)

def main(args):
    cfg = yaml.safe_load(open(args.config))
    set_all_seeds(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"])
    out_dir = cfg.get("out_dir", "./runs"); os.makedirs(out_dir, exist_ok=True)
    slices_dir = os.path.join(out_dir, "slices"); os.makedirs(slices_dir, exist_ok=True)

    classes = tuple(cfg["data"]["classes"])
    target_shape = tuple(cfg["data"]["target_shape"])
    ds_test = MRIVolumeDataset(cfg["data"]["test_dir"], classes, target_shape, train=False)

    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False,
                         num_workers=cfg["train"]["num_workers"], pin_memory=False)

    model = CNN3D(cfg["model"]["in_ch"], cfg["model"]["num_classes"], cfg["model"]["dropout"]).to(device)
    state = torch.load(cfg["test"]["load_path"], map_location=device)
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    model.load_state_dict(state); model.eval()

    th_cfg = cfg["test"].get("threshold", 0.5)
    th = _load_auto_threshold(out_dir) if str(th_cfg).lower() == "auto" else float(th_cfg)

    y_true, y_pred, y_prob, paths = [], [], [], []
    axial_slices, coronal_slices, sagittal_slices = [], [], []
    preds_for_grid, labels_for_grid = [], []

    with torch.no_grad():
        for x, y, p in dl_test:
            x = x.to(device, non_blocking=True).float()
            prob_ad = torch.softmax(model(x), dim=1)[:, 1].cpu().item()
            pred = int(prob_ad >= th)

            y_true.append(int(y.item()))
            y_pred.append(pred)
            y_prob.append(prob_ad)
            paths.append(p[0])

            vol = load_canonical(p[0]); vol = normalize_clip_z(vol); vol = resize_3d(vol, target_shape)

            if cfg["test"].get("save_slices", True):
                title = f"pred={pred}  true={int(y.item())}  P(AD)={prob_ad:.2f}"
                idx = (cfg["test"].get("slice_indices") or ["center"])[0]
                out_img = os.path.join(slices_dir, os.path.basename(p[0]).replace(".nii.gz","_slices.png").replace(".nii","_slices.png"))
                save_three_orthogonal_slices(vol, out_img, title=title, cmap=cfg["test"].get("cmap","gray"), idx=idx)

            d, h, w = vol.shape
            axial_slices.append(vol[d//2, :, :])
            coronal_slices.append(vol[:, h//2, :])
            sagittal_slices.append(vol[:, :, w//2])
            preds_for_grid.append(pred); labels_for_grid.append(int(y.item()))

    m = compute_basic_metrics(np.array(y_true), np.array(y_pred))
    print(f"Accuracy: {m['accuracy']*100:.2f}%")
    print("Confusion matrix (raw counts):")
    print(m["confusion_matrix"])

    plot_confusion_matrix(m["confusion_matrix"], os.path.join(out_dir, "confusion_matrix.png"),
                          class_names=classes)

    with open(os.path.join(out_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path","true","pred","prob_ad"])
        for pth, yt, yp, pr in zip(paths, y_true, y_pred, y_prob):
            w.writerow([pth, yt, yp, f"{pr:.6f}"])

    grid_cols = 5
    save_slice_grid(axial_slices, preds_for_grid, labels_for_grid, os.path.join(slices_dir, "grid_axial.png"), cols=grid_cols, cmap=cfg["test"].get("cmap","gray"), title="Axial Slices")
    save_slice_grid(coronal_slices, preds_for_grid, labels_for_grid, os.path.join(slices_dir, "grid_coronal.png"), cols=grid_cols, cmap=cfg["test"].get("cmap","gray"), title="Coronal Slices")
    save_slice_grid(sagittal_slices, preds_for_grid, labels_for_grid, os.path.join(slices_dir, "grid_sagittal.png"), cols=grid_cols, cmap=cfg["test"].get("cmap","gray"), title="Sagittal Slices")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args())
