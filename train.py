import os, argparse, yaml, inspect
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import MRIVolumeDataset
from src.models.cnn3d import CNN3D
from src.utils.seed import set_all_seeds
from src.vis.plots import plot_train_loss


def build_model(cfg):
    return CNN3D(cfg["model"]["in_ch"], cfg["model"]["num_classes"], cfg["model"]["dropout"])

def _class_counts(ds):
    c0 = sum(1 for _, y, _ in ds if int(y) == 0)
    c1 = sum(1 for _, y, _ in ds if int(y) == 1)
    return c0, c1

def _compute_class_weights(ds, device):
    c0, c1 = _class_counts(ds)
    counts = torch.tensor([c0, c1], dtype=torch.float32, device=device)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * 2.0
    return weights

def _make_balanced_sampler(ds):
    c0, c1 = _class_counts(ds)
    w0, w1 = (1.0 / (c0 + 1e-6)), (1.0 / (c1 + 1e-6))
    weights = [w0 if int(y) == 0 else w1 for _, y, _ in ds]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def _find_best_threshold(probs, labels):
    labels = np.asarray(labels); probs = np.asarray(probs)
    best_t, best_bacc = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (probs >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tpr = tp / max(1, (tp + fn))
        tnr = tn / max(1, (tn + fp))
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc, best_t = bacc, float(t)
    return best_t, best_bacc

def main(args):
    cfg = yaml.safe_load(open(args.config))
    set_all_seeds(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"])
    out_dir = cfg.get("out_dir", "./runs"); os.makedirs(out_dir, exist_ok=True)

    # train_ds = MRIVolumeDataset(root_dir=cfg["data"]["train_dir"],classes=tuple(cfg["data"]["classes"]),target_shape=tuple(cfg["data"]["target_shape"]),
    #     train=True,augment_cfg=cfg["data"]["augment"],
    # )
    train_ds = MRIVolumeDataset(
    root_dir=cfg["data"]["train_dir"],
    classes=tuple(cfg["data"]["classes"]),
    target_shape=tuple(cfg["data"]["target_shape"]),
    train=True,
    augment_cfg=cfg["data"]["augment"],
    preproc_cfg=cfg["data"]["preproc"],   # <<< add this
)
    print("Train counts (health, patient):", _class_counts(train_ds))

    sampler = _make_balanced_sampler(train_ds)
    dl_train = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                          sampler=sampler, num_workers=cfg["train"]["num_workers"], pin_memory=False)

    model = build_model(cfg).to(device)

    if cfg["train"].get("class_weighting", "auto") == "auto":
        class_w = _compute_class_weights(train_ds, device)
        try:
            criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.02)
        except TypeError:
            criterion = nn.CrossEntropyLoss(weight=class_w)
    else:
        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
        except TypeError:
            criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    sched_args = dict(mode='min', factor=0.5, patience=2, threshold=1e-4, min_lr=1e-6, cooldown=1)
    if "verbose" in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters:
        sched_args["verbose"] = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_args)

    epochs = cfg["train"]["epochs"]
    acc_steps = int(cfg["train"].get("accumulate_steps", 1))
    best_running = float("inf")
    best_path = os.path.join(out_dir, "best_model.pt")
    train_losses = []

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, nb = 0.0, 0
        for x, y, _ in dl_train:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)

            if step % acc_steps == 0:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y) / acc_steps
            loss.backward()

            if (step % acc_steps) == (acc_steps - 1):
                if cfg["train"]["grad_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                optimizer.step()

            run_loss += loss.item() * acc_steps
            nb += 1; step += 1

        epoch_loss = run_loss / max(1, nb)
        train_losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch:03d} | train_loss={epoch_loss:.4f}")

        if epoch_loss < best_running - 1e-6:
            best_running = epoch_loss
            torch.save(model.state_dict(), best_path)

    plot_train_loss(train_losses, os.path.join(out_dir, "loss.png"))
    print(f"Best model saved to: {best_path}")

    # # ---- learn auto threshold on the same subjects (no aug) ----
    # train_eval = MRIVolumeDataset(
    #     root_dir=cfg["data"]["train_dir"],
    #     classes=tuple(cfg["data"]["classes"]),
    #     target_shape=tuple(cfg["data"]["target_shape"]),
    #     train=False,
    #     augment_cfg=cfg["data"]["augment"],
    # )

    train_eval = MRIVolumeDataset(
    root_dir=cfg["data"]["train_dir"],
    classes=tuple(cfg["data"]["classes"]),
    target_shape=tuple(cfg["data"]["target_shape"]),
    train=False,
    augment_cfg=cfg["data"]["augment"],
    preproc_cfg=cfg["data"]["preproc"],   # <<< add this
)

    dl_eval = DataLoader(train_eval, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y, _ in dl_eval:
            x = x.to(device).float()
            pr = torch.softmax(model(x), dim=1)[:, 1].cpu().item()
            probs.append(pr); labels.append(int(y.item()))
    t_star, bacc = _find_best_threshold(np.array(probs), np.array(labels))
    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(f"{t_star:.4f}\n")
    print(f"Saved auto threshold to runs/threshold.txt (best balanced acc {bacc:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args())
