import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate
import nibabel as nib
from .registrations import register_volume_to_mni, load_canonical_img, img_to_array


# --------- light augmentations ----------
def robust_norm(vol, p_lo=1, p_hi=99):
    lo, hi = np.percentile(vol, [p_lo, p_hi])
    vol = np.clip(vol, lo, hi)
    m, s = vol.mean(), vol.std()
    return (vol - m) / (s + 1e-8)

def center_crop_or_pad(vol, target=(128,128,128), pad_value=0.0):
    d, h, w = vol.shape
    td, th, tw = target
    out = np.full(target, pad_value, dtype=vol.dtype)

    sd = max(0, (d - td)//2); sh = max(0, (h - th)//2); sw = max(0, (w - tw)//2)
    ed = min(d, sd + td); eh = min(h, sh + th); ew = min(w, sw + tw)

    dd = max(0, (td - d)//2); dh = max(0, (th - h)//2); dw = max(0, (tw - w)//2)

    out[dd:dd+(ed-sd), dh:dh+(eh-sh), dw:dw+(ew-sw)] = vol[sd:ed, sh:eh, sw:ew]
    return out

def resize_3d(vol, target=(128,128,128)):
    factors = [t / s for t, s in zip(target, vol.shape)]
    return zoom(vol, zoom=factors, order=1)


def random_flip3d(vol, p=0.5):
    import random
    if random.random() < p: vol = vol[::-1, :, :].copy()
    if random.random() < p: vol = vol[:, ::-1, :].copy()
    if random.random() < p: vol = vol[:, :, ::-1].copy()
    return vol

def random_rotate_small(vol, max_deg=7):
    import random
    axes = random.choice([(0,1), (0,2), (1,2)])
    deg = random.uniform(-max_deg, max_deg)
    return rotate(vol, angle=deg, axes=axes, reshape=False, order=1, mode='nearest')

def random_gamma(vol, jitter=0.1):
    import random
    g = 1.0 + random.uniform(-jitter, jitter)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-6: return vol
    v01 = (vol - vmin) / (vmax - vmin + 1e-8)
    return (np.power(v01, g) * (vmax - vmin) + vmin)

def add_noise(vol, std=0.01):
    return vol + np.random.normal(0.0, std, size=vol.shape).astype(np.float32)

# --------- dataset ----------

class MRIVolumeDataset(Dataset):
    """
    root_dir/
      health/*.nii*
      patient/*.nii*
    """
    def __init__(self, root_dir, classes=('health','patient'),
                 target_shape=(128,128,128),
                 train=True,
                 augment_cfg=None,
                 file_list=None,
                 preproc_cfg=None):
        super().__init__()
        self.root_dir = root_dir
        self.classes = list(classes)
        self.cls2id = {c: i for i, c in enumerate(self.classes)}
        self.target = tuple(target_shape)
        self.train = train
        self.augment_cfg = augment_cfg or {}
        self.preproc = preproc_cfg or {}

        if file_list is None:
            files = []
            for c in self.classes:
                pat = os.path.join(root_dir, c, "*.nii*")
                for p in sorted(glob.glob(pat)):
                    files.append((p, self.cls2id[c]))
        else:
            files = file_list
        self.samples = files

        self.cache_dir = self.preproc.get("cache_dir", None)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ---------- core I/O with registration + cache ----------

    def cache_path(self, nifti_path: str) -> str:
        base = os.path.basename(nifti_path)
        iso = int(self.preproc.get("mni_iso_mm", 2))
        msk = int(bool(self.preproc.get("apply_brain_mask", True)))
        key = f"{base}|{iso}|{msk}"
        import hashlib
        f = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, f + ".npy")

    def load_registered(self, path: str) -> np.ndarray:
        use_mni = bool(self.preproc.get("register_to_mni", False))
        iso_mm = int(self.preproc.get("mni_iso_mm", 2))
        apply_mask = bool(self.preproc.get("apply_brain_mask", True))

        if use_mni:
            if self.cache_dir:
                cp = self.cache_path(path)
                if os.path.exists(cp):
                    return np.load(cp)
                vol = register_volume_to_mni(path, iso_mm=iso_mm, apply_brain_mask=apply_mask)
                np.save(cp, vol.astype(np.float32))
                return vol.astype(np.float32)
            else:
                return register_volume_to_mni(path, iso_mm=iso_mm, apply_brain_mask=apply_mask).astype(np.float32)
        else:
            # fallback: canonical reorientation only
            img = load_canonical_img(path)
            return img_to_array(img).astype(np.float32)

    # ---------- public ----------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        vol = self.load_registered(path)

        # intensity norm
        vol = robust_norm(vol)

        # unify to model grid (MNI 2mm ~ 91x109x91; we crop/pad or upscale to 128³)
        vol = center_crop_or_pad(vol, target=self.target)

        # TRAIN-time light augs
        if self.train and self.augment_cfg.get("enable", True):
            if self.augment_cfg.get("flip_prob", 0) > 0:
                vol = random_flip3d(vol, p=self.augment_cfg.get("flip_prob", 0.5))
            if self.augment_cfg.get("max_rotate_deg", 0) > 0:
                vol = random_rotate_small(vol, max_deg=self.augment_cfg.get("max_rotate_deg", 7))
            if self.augment_cfg.get("gamma_jitter", 0) > 0:
                vol = random_gamma(vol, jitter=self.augment_cfg.get("gamma_jitter", 0.1))
            if self.augment_cfg.get("noise_std", 0) > 0:
                vol = add_noise(vol, std=self.augment_cfg.get("noise_std", 0.01))
            vol = robust_norm(vol)

        x = torch.from_numpy(vol).unsqueeze(0)  # [1,D,H,W]
        y = torch.tensor(y, dtype=torch.long)
        return x, y, path
