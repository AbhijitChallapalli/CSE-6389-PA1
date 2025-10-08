import os
import glob
import random
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, rotate
import torch
from torch.utils.data import Dataset

# --- I/O & preprocessing ---

def load_canonical(path):
    """
    Load NIfTI and reorient to closest canonical (RAS-like)
    """
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata().astype(np.float32)
    return vol

def normalize_clip_z(vol, p_lo=1, p_hi=99):
    """
    Robust per-volume normalization:
      1) clip to [p_lo, p_hi] percentile
      2) z-score
    """
    lo, hi = np.percentile(vol, [p_lo, p_hi])
    vol = np.clip(vol, lo, hi)
    m, s = vol.mean(), vol.std()
    return (vol - m) / (s + 1e-8)

def resize_3d(vol, target=(128, 128, 128)):
    """
    Trilinear resample to target (D,H,W).
    """
    factors = [t / s for t, s in zip(target, vol.shape)]
    return zoom(vol, zoom=factors, order=1)

# --- light augmentations for tiny data ---

def random_flip3d(vol, p=0.5):
    if random.random() < p: vol = vol[::-1, :, :].copy()
    if random.random() < p: vol = vol[:, ::-1, :].copy()
    if random.random() < p: vol = vol[:, :, ::-1].copy()
    return vol

def random_rotate_small(vol, max_deg=7):
    axes = random.choice([(0,1), (0,2), (1,2)])
    deg = random.uniform(-max_deg, max_deg)
    return rotate(vol, angle=deg, axes=axes, reshape=False, order=1, mode='nearest')

def random_gamma(vol, jitter=0.1):
    g = 1.0 + random.uniform(-jitter, jitter)
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-6: return vol
    v01 = (vol - vmin) / (vmax - vmin + 1e-8)
    v01 = np.power(v01, g)
    return v01 * (vmax - vmin) + vmin

def add_noise(vol, std=0.01):
    return vol + np.random.normal(0.0, std, size=vol.shape).astype(np.float32)

# --- Dataset ---

class MRIVolumeDataset(Dataset):
    """
    Dataset expects root_dir with subdirs named by classes (e.g., health/patient)
    Each folder contains subject volumes as .nii or .nii.gz
    """
    def __init__(self, root_dir, classes=('health','patient'), target_shape=(128,128,128),
                 train=True, augment_cfg=None, file_list=None):
        super().__init__()
        self.root_dir = root_dir
        self.classes = list(classes)
        self.cls2id = {c: i for i, c in enumerate(self.classes)}
        self.target = tuple(target_shape)
        self.train = train
        self.augment_cfg = augment_cfg or {}

        if file_list is None:
            files = []
            for c in self.classes:
                pat = os.path.join(root_dir, c, "*.nii*")
                for p in sorted(glob.glob(pat)):
                    files.append((p, self.cls2id[c]))
        else:
            files = file_list

        self.samples = files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        # load & preprocess
        vol = load_canonical(path)
        vol = normalize_clip_z(vol)               # robust norm first
        vol = resize_3d(vol, self.target)         # resample to fixed shape

        if self.train and self.augment_cfg.get("enable", True):
            vol = random_flip3d(vol, p=self.augment_cfg.get("flip_prob", 0.5))
            vol = random_rotate_small(vol, max_deg=self.augment_cfg.get("max_rotate_deg", 7))
            vol = random_gamma(vol, jitter=self.augment_cfg.get("gamma_jitter", 0.1))
            vol = add_noise(vol, std=self.augment_cfg.get("noise_std", 0.01))
            vol = normalize_clip_z(vol)
            vol = vol.astype(np.float32, copy=False)  


        # [C,D,H,W] for 3D conv
        x = torch.from_numpy(vol).unsqueeze(0)    # [1,D,H,W], float32
        y = torch.tensor(y, dtype=torch.long)
        return x, y, path
