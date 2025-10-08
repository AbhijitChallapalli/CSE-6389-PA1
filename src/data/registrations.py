# src/data/registrations.py
import hashlib
import numpy as np
import nibabel as nib
from nilearn import datasets as nl_datasets
from nilearn.image import resample_to_img

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def load_canonical_img(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img  

def img_to_array(img) -> np.ndarray:
    return img.get_fdata().astype(np.float32)

_MNI_CACHE = {}

def get_mni_template_and_mask(iso_mm=2, want_mask=True):
    """
    Load MNI152 template (and optional brain mask) from nilearn local cache.
    iso_mm: 1 or 2 (millimeters).
    """
    key = f"{iso_mm}_{int(bool(want_mask))}"
    if key in _MNI_CACHE:
        return _MNI_CACHE[key]
    tmpl = nl_datasets.load_mni152_template(resolution=iso_mm)   # Nifti1Image
    mask = nl_datasets.load_mni152_brain_mask(resolution=iso_mm) if want_mask else None
    _MNI_CACHE[key] = (tmpl, mask)
    return _MNI_CACHE[key]

def register_volume_to_mni(path, iso_mm=2, apply_brain_mask=True):
    """
    Affine registration to MNI template grid using nilearn.resample_to_img.
    Returns a float32 numpy array in MNI space.
    """
    sub_img = load_canonical_img(path)
    tmpl, mask = get_mni_template_and_mask(iso_mm=iso_mm, want_mask=apply_brain_mask)
    sub_to_mni_img = resample_to_img(source_img=sub_img, target_img=tmpl, interpolation="continuous")
    vol = img_to_array(sub_to_mni_img)
    if apply_brain_mask and mask is not None:
        mask_data = img_to_array(mask).astype(bool)
        vol = vol * mask_data
    return vol
