import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
import json
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import os


# This python file includes constants for use in multiple locations, 
# custom dataset definition, utility methods for working with images - crop, plot,
# utility methods to process data and save reusable structures, methods for running 
# inference and calculating metrics for models evaluation.


# CONSTANTS used across different notebooks:

LABEL_COLUMNS = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
    "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]

LABEL_NAMES = {
    "AKIEC": "Actinic keratosis / intraepidermal carcinoma",
    "BCC": "Basal cell carcinoma",
    "BEN_OTH": "Other benign proliferations",
    "BKL": "Benign keratinocytic lesion",
    "DF": "Dermatofibroma",
    "INF": "Inflammatory / infectious",
    "MAL_OTH": "Other malignant proliferations",
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "SCCKA": "Squamous cell carcinoma / keratoacanthoma",
    "VASC": "Vascular lesions / hemorrhage"
}

BENIGNANT_NAMES = {
    "BEN_OTH",
    "BKL",
    "NV",
    "VASC",
    "DF",
    "INF"
}

MALIGNANT_NAMES = {
    "MAL_OTH",
    "AKIEC",
    "BCC",
    "MEL",
    "SCCKA"
}

########### CUSTOM DATASETS:

# 1. Unimodal: for one image type - clinical close-up or dermoscopic.

class Milk10kDataset_unimodal(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.root_dir / row['lesion_id'] / f"{row['isic_id']}.jpg"
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = row["label_id"]

        return img, label


# 2. Multimodal Dataset: Image + Metadata - one image type and custom metadata columns

# metadata_maps make sure that the one-hot encoding of the metadata column values
# is always done in the same predictable way for training, validation, test.

# Smoothing creates soft-one-hot encoding.

class Milk10kDataset_multimodal(Dataset):
    def __init__(
        self,
        df,
        root_dir,
        metadata_maps,  
        metadata_cols=None,
        transform=None,
        smoothing=0.1,
        nan_placeholder="MISSING"
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.smoothing = smoothing
        self.metadata_maps = metadata_maps
        self.nan_placeholder = nan_placeholder

        if metadata_cols is None:
            metadata_cols = list(metadata_maps.keys())
        self.metadata_cols = metadata_cols

        # Fixed metadata dimension (sum of one-hot lengths)
        self.metadata_dim = sum(len(mapping) for mapping in metadata_maps.values())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image 
        img_path = self.root_dir / row['lesion_id'] / f"{row['isic_id']}.jpg"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Metadata 
        metadata_vec = torch.full((self.metadata_dim,), self.smoothing / 2.0)
        offset = 0

        for col in self.metadata_cols:
            val = row[col]
            if pd.isna(val):
                val = self.nan_placeholder
            val = str(val)  # convert to string to match metadata_maps keys

            col_map = self.metadata_maps[col]
            if val in col_map:
                idx_in_col = col_map[val]
                metadata_vec[offset + idx_in_col] = 1.0 - self.smoothing
            offset += len(col_map)

        label = int(row["label_id"])
        return img, metadata_vec, label
        
        
# Version that zeroes-out given samples by lession Id for given features:
class Milk10kDataset_multimodal_feature_unlearning(Dataset):
    def __init__(
        self,
        df,
        root_dir,
        metadata_maps,
        metadata_cols=None,
        transform=None,
        smoothing=0.1,
        nan_placeholder="MISSING",
        forget_lesion_ids=None,      
        forget_metadata_cols=None    
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.smoothing = smoothing
        self.metadata_maps = metadata_maps
        self.nan_placeholder = nan_placeholder
        self.forget_lesion_ids = set(forget_lesion_ids or [])       
        self.forget_metadata_cols = set(forget_metadata_cols or []) 
        if metadata_cols is None:
            metadata_cols = list(metadata_maps.keys())
        self.metadata_cols = metadata_cols
        self.metadata_dim = sum(len(mapping) for mapping in metadata_maps.values())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Image
        img_path = self.root_dir / row['lesion_id'] / f"{row['isic_id']}.jpg"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Metadata
        metadata_vec = torch.full((self.metadata_dim,), self.smoothing / 2.0)
        offset = 0
        for col in self.metadata_cols:
            val = row[col]
            if pd.isna(val):
                val = self.nan_placeholder
            val = str(val)
            col_map = self.metadata_maps[col]
            if val in col_map:
                idx_in_col = col_map[val]
                if (                                                  
                    row["lesion_id"] in self.forget_lesion_ids        
                    and col in self.forget_metadata_cols              
                ):                                                    
                    metadata_vec[offset + idx_in_col] = self.smoothing / 2.0  
                else:
                    metadata_vec[offset + idx_in_col] = 1.0 - self.smoothing
            offset += len(col_map)
        label = int(row["label_id"])
        return img, metadata_vec, label

 
# 3. Multimodal Dataset: Image + Image - both image types, no metadata
class Milk10kDataset_images(Dataset):
    def __init__(
        self,
        df_clinical,
        df_dermoscopic,
        root_dir,
        transform=None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Merge on lesion_id 
        self.df = df_clinical.merge(
            df_dermoscopic,
            on="lesion_id",
            suffixes=("_clin", "_derm")
        ).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image 1: clinical 
        img1_path = self.root_dir / row['lesion_id'] / f"{row['isic_id_clin']}.jpg"
        img1 = Image.open(img1_path).convert("RGB")

        # Image 2: dermoscopic
        img2_path = self.root_dir / row['lesion_id'] / f"{row['isic_id_derm']}.jpg"
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = int(row["label_id_clin"])  # same label for both

        return img1, img2, label
        
# 4. Multimodal dataset: Image + Image + metadata

# both clinical close-up and dermoscopic image
# metadata_maps make sure that the one-hot encoding of the metadata column values
# is always done in the same predictable way for training, validation, test.
# Smoothing creates soft-one-hot encoding.

class Milk10kDataset_3_modalities(Dataset):
    def __init__(
        self,
        df_clinical,
        df_dermoscopic,
        root_dir,
        metadata_maps,
        metadata_cols=None,
        transform=None,
        smoothing=0.1,
        nan_placeholder="MISSING"
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.smoothing = smoothing
        self.metadata_maps = metadata_maps
        self.nan_placeholder = nan_placeholder

        if metadata_cols is None:
            metadata_cols = list(metadata_maps.keys())
        self.metadata_cols = metadata_cols

        # Merge clinical + dermoscopic
        self.df = df_clinical.merge(
            df_dermoscopic,
            on="lesion_id",
            suffixes=("_clin", "_derm")
        ).reset_index(drop=True)

        # Metadata dimension
        self.metadata_dim = sum(len(mapping) for mapping in metadata_maps.values())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # IMAGE 1 (clinical) 
        img1_path = self.root_dir / row['lesion_id'] / f"{row['isic_id_clin']}.jpg"
        img1 = Image.open(img1_path).convert("RGB")

        # IMAGE 2 (dermoscopic)
        img2_path = self.root_dir / row['lesion_id'] / f"{row['isic_id_derm']}.jpg"
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # METADATA (from clinical row) 
        #metadata_vec = torch.full((self.metadata_dim,), self.smoothing)
        metadata_vec = torch.full((self.metadata_dim,), self.smoothing / 2.0)

        offset = 0
        for col in self.metadata_cols:
            val = row[f"{col}_clin"] if f"{col}_clin" in row else row[col]

            if pd.isna(val):
                val = self.nan_placeholder

            val = str(val)

            col_map = self.metadata_maps[col]
            if val in col_map:
                idx_in_col = col_map[val]
                #metadata_vec[offset + idx_in_col] = 1.0
                metadata_vec[offset + idx_in_col] = 1.0 - self.smoothing

            offset += len(col_map)

        # LABEL 
        label = int(row["label_id_clin"])  # same as derm

        return img1, img2, metadata_vec, label
        
########## Feature unlearning dataset:

class Milk10kDataset_3_modalities_feature_unlearning(Dataset):
    def __init__(
        self,
        df_clinical,
        df_dermoscopic,
        root_dir,
        metadata_maps,
        metadata_cols=None,
        transform=None,
        forget_lesion_ids=None,
        forget_metadata_cols=None,
        smoothing=0.1,
        nan_placeholder="MISSING"
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.smoothing = smoothing
        self.metadata_maps = metadata_maps
        self.nan_placeholder = nan_placeholder
        self.forget_lesion_ids = set(forget_lesion_ids or [])
        self.forget_metadata_cols = set(forget_metadata_cols or [])

        if metadata_cols is None:
            metadata_cols = list(metadata_maps.keys())
        self.metadata_cols = metadata_cols

        # Merge clinical + dermoscopic
        self.df = df_clinical.merge(
            df_dermoscopic,
            on="lesion_id",
            suffixes=("_clin", "_derm")
        ).reset_index(drop=True)

        # Metadata dimension
        self.metadata_dim = sum(len(mapping) for mapping in metadata_maps.values())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # IMAGE 1 (clinical) 
        img1_path = self.root_dir / row['lesion_id'] / f"{row['isic_id_clin']}.jpg"
        img1 = Image.open(img1_path).convert("RGB")

        # IMAGE 2 (dermoscopic)
        img2_path = self.root_dir / row['lesion_id'] / f"{row['isic_id_derm']}.jpg"
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # METADATA (from clinical row) 
        #metadata_vec = torch.full((self.metadata_dim,), self.smoothing)
        metadata_vec = torch.full((self.metadata_dim,), self.smoothing / 2.0)

        offset = 0
        for col in self.metadata_cols:
            val = row[f"{col}_clin"] if f"{col}_clin" in row else row[col]

            if pd.isna(val):
                val = self.nan_placeholder

            val = str(val)

            col_map = self.metadata_maps[col]
        
            if val in col_map:
                idx_in_col = col_map[val]
                if (
                    row["lesion_id"] in self.forget_lesion_ids
                    and col in self.forget_metadata_cols
                ):
                    #metadata_vec[offset + idx_in_col] = self.smoothing
                    metadata_vec[offset + idx_in_col] = self.smoothing / 2.0
                else:
                    #metadata_vec[offset + idx_in_col] = 1.0
                    metadata_vec[offset + idx_in_col] = 1.0 - self.smoothing

            offset += len(col_map)

        # LABEL 
        label = int(row["label_id_clin"])  # same as derm
        return img1, img2, metadata_vec, label

########## UTILITY METHODS:

# Saving and loading data for reuse in different notebooks and between restart of Colab sessions:


def save_lesion_split_json(train_df, val_df, test_df, out_path):
    split = {
        "train": sorted(train_df["lesion_id"].unique().tolist()),
        "val": sorted(val_df["lesion_id"].unique().tolist()),
        "test": sorted(test_df["lesion_id"].unique().tolist()),
    }

    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Saved {out_path}")
    print(
        f"Lesions - Train: {len(split['train'])}, "
        f"Val: {len(split['val'])}, "
        f"Test: {len(split['test'])}"
    )


def load_lesion_splits(json_path):
    with open(json_path, "r") as f:
        split = json.load(f)

    train_lesions = set(split["train"])
    val_lesions   = set(split["val"])
    test_lesions  = set(split["test"])

    return train_lesions, val_lesions, test_lesions
    

def process_and_save_images(
    df,
    source_root,
    target_root,
    train_lesions,
    val_lesions,
    test_lesions,
    image_size=224
):
    source_root = Path(source_root)
    target_root = Path(target_root)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        lesion_id = row["lesion_id"]
        isic_id = row["isic_id"]
        image_type = row["image_type"]

        # Determine split
        if lesion_id in train_lesions:
            split = "train"
        elif lesion_id in val_lesions:
            split = "validation"
        elif lesion_id in test_lesions:
            split = "test"
        else:
            # Should not happen, but safe to skip
            continue

        # Source image path
        src_path = source_root / lesion_id / f"{isic_id}.jpg"
        if not src_path.exists():
            print(f"Missing image: {src_path}")
            continue

        # Load image
        img = Image.open(src_path).convert("RGB")

        # Crop → Resize
        img = center_square_crop(img)
        img = img.resize((image_size, image_size), Image.BILINEAR)

        # Target directory
        dst_dir = target_root / split / lesion_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Save image (keep original filename)
        dst_path = dst_dir / f"{isic_id}.jpg"
        img.save(dst_path, quality=95)
        
        
def build_metadata_maps(df, metadata_cols, nan_placeholder="MISSING"):
    """
    Build one-hot index maps for metadata columns.
    Converts all keys to strings for JSON safety.
    NaNs are replaced with nan_placeholder.
    Values are sorted to ensure consistent ordering.
    """
    metadata_maps = {}
    for col in metadata_cols:
        unique_vals = df[col].fillna(nan_placeholder).unique()
        # Convert all values to strings
        unique_vals = [str(v) for v in unique_vals]
        # Sort the values for consistent ordering
        unique_vals = sorted(unique_vals)
        metadata_maps[col] = {v: i for i, v in enumerate(unique_vals)}
    return metadata_maps


FORGET_SIZE = 225
WORK_DIR_ROOT = '/content/drive/MyDrive/datasets/Master/'


def create_unlearning_split_youth(young_ratio, split_filename, total_forget=FORGET_SIZE, random_seed=67):
    """
    Creates or loads an unlearning split where:
    - young_ratio % of total_forget comes from age_approx <= 30
    - (1 - young_ratio) % comes randomly from the rest

    Args:
        young_ratio: float, e.g. 0.05 for 5%
        split_filename: str, e.g. "unlearning_lesion_split_age_5_ran_95.json"
        total_forget: int, total number of forget lesions (default 209)
        random_seed: int (default 67)
    """
    
    split_path = WORK_DIR_ROOT + "/" + split_filename

    if not os.path.exists(split_path):
        print(f"Creating new unlearning split: {split_filename}")

        random.seed(random_seed)

        n_young = int(total_forget * young_ratio)
        n_other = total_forget - n_young

        young_lesion_ids = set(
            df_train_clinical_minimal[
                df_train_clinical_minimal["age_approx"] <= 30
            ]["lesion_id"].unique()
        )

        young_train = [l for l in train_lesions if l in young_lesion_ids]
        other_train = [l for l in train_lesions if l not in young_lesion_ids]

        print(f"  Young lesions available in train: {len(young_train)}")
        print(f"  Other lesions available in train: {len(other_train)}")

        if len(young_train) < n_young:
            raise ValueError(f"Not enough young lesions: need {n_young}, have {len(young_train)}")
        if len(other_train) < n_other:
            raise ValueError(f"Not enough other lesions: need {n_other}, have {len(other_train)}")

        forget_young = random.sample(young_train, n_young)
        forget_other = random.sample(other_train, n_other)

        forget_lesions = forget_young + forget_other
        retain_lesions = [l for l in train_lesions if l not in set(forget_lesions)]

        print(f"  Forget: {len(forget_lesions)} ({n_young} young + {n_other} other)")
        print(f"  Retain: {len(retain_lesions)}")

        unlearning_split = {
            "forget": forget_lesions,
            "retain": retain_lesions
        }

        with open(split_path, "w") as f:
            json.dump(unlearning_split, f, indent=2)

    else:
        print(f"Loading existing split: {split_filename}")

        with open(split_path, "r") as f:
            unlearning_split = json.load(f)

        forget_lesions = unlearning_split["forget"]
        retain_lesions = unlearning_split["retain"]
        print(f"  Forget: {len(forget_lesions)} | Retain: {len(retain_lesions)}")

    return forget_lesions, retain_lesions
    
# Methods to crop, plot images:

def center_square_crop(img: Image.Image) -> Image.Image:
    """
    Crop the largest possible square from the center of the image.
    """
    w, h = img.size
    side = min(w, h)

    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side

    return img.crop((left, top, right, bottom))


def plot_transformed_images(
    dataset_root,
    split_type,
    transform,
    n=3,
    seed=42
):
    random.seed(seed)

    dataset_root = Path(dataset_root)
    split_dir = dataset_root / split_type

    # 1️ get lesion folders (FAST)
    lesion_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    if len(lesion_dirs) == 0:
        raise RuntimeError(f"No lesion folders found in {split_dir}")

    # 2️ sample ONLY n lesions
    sampled_lesions = random.sample(
        lesion_dirs, min(n, len(lesion_dirs))
    )

    for lesion_dir in sampled_lesions:
        # 3️ pick ONE image from that lesion
        image_paths = list(lesion_dir.glob("*.jpg"))
        if not image_paths:
            continue

        img_path = random.choice(image_paths)

        # 4️ load image
        img = Image.open(img_path).convert("RGB")

        # 5️ apply transform ONLY HERE
        transformed = transform(img)
        if isinstance(transformed, torch.Tensor):
            transformed = transformed.permute(1, 2, 0)

        # 6️ plot
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img)
        ax[0].set_title(f"Original\n{img.size}")
        ax[0].axis("off")

        ax[1].imshow(transformed)
        ax[1].set_title("Transformed")
        ax[1].axis("off")

        fig.suptitle(
            f"{split_type} | {lesion_dir.name} | {img_path.name}",
            fontsize=10
        )

        plt.show()


# Methods for evaluating and testing models performance:

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_targets).numpy()
    )


@torch.no_grad()
def top_k_accuracy(model, loader, device, k=3):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        topk = outputs.topk(k, dim=1).indices

        correct += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    return correct / total


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)   # ONE forward pass
        all_logits.append(outputs.cpu())
        all_targets.append(labels.cpu())

    return (
        torch.cat(all_logits),   # shape [N, num_classes]
        torch.cat(all_targets)   # shape [N]
    )
    
    
@torch.no_grad()
def run_inference_multimodal(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []

    for images, meta, labels in loader:
        images = images.to(device, non_blocking=True)
        meta   = meta.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images, meta)   # ONE forward pass
        all_logits.append(outputs.cpu())
        all_targets.append(labels.cpu())

    return (
        torch.cat(all_logits),   # shape [N, num_classes]
        torch.cat(all_targets)   # shape [N]
    )


@torch.no_grad()
def run_inference_images(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []

    for image_1, image_2, labels in loader:
        image_1 = image_1.to(device, non_blocking=True)
        image_2 = image_2.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        outputs = model(image_1, image_2)  # forward pass

        all_logits.append(outputs.cpu())
        all_targets.append(labels.cpu())

    return (
        torch.cat(all_logits),   # [N, num_classes]
        torch.cat(all_targets)   # [N]
    )


@torch.no_grad()
def run_inference_3_modalities(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []

    for image_1, image_2, metadata, labels in loader:
        image_1 = image_1.to(device, non_blocking=True)
        image_2 = image_2.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        outputs = model(image_1, image_2, metadata)  # forward pass

        all_logits.append(outputs.cpu())
        all_targets.append(labels.cpu())

    return (
        torch.cat(all_logits),   # [N, num_classes]
        torch.cat(all_targets)   # [N]
    )

def top_k_from_logits(logits, targets, k=3):
    topk = logits.topk(k, dim=1).indices
    return (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()

########### MODEL EVALUATION METRICS


# Mean square error between the logits of the forget set in the reference and unlearned model, 
# should be close to 0 in ideal case

def compute_forgetting_mse(reference_model, unlearned_model, dataloader, device, num_classes=11):
    reference_model.eval()
    unlearned_model.eval()
    mse_list = []
    with torch.no_grad():
        for batch in dataloader:
            image_1, image_2, metadata, labels = [x.to(device) for x in batch]
            logits_ref = reference_model(image_1, image_2, metadata)
            logits_unl = unlearned_model(image_1, image_2, metadata)
            probs_ref = F.softmax(logits_ref, dim=1)
            probs_unl = F.softmax(logits_unl, dim=1)
            mse = torch.mean((probs_ref - probs_unl) ** 2, dim=1)
            mse_list.extend(mse.cpu().numpy())

    raw_mse = sum(mse_list) / len(mse_list)
    max_mse = 1.0 / num_classes
    return raw_mse / max_mse

# Version for only metadata and image

import torch
import torch.nn.functional as F

def compute_forgetting_mse_image_metadata(
    reference_model,
    unlearned_model,
    dataloader,
    device,
    num_classes=11
):
    reference_model.eval()
    unlearned_model.eval()

    mse_list = []

    with torch.no_grad():
        for batch in dataloader:
            image, metadata, labels = [x.to(device) for x in batch]

            logits_ref = reference_model(image, metadata)
            logits_unl = unlearned_model(image, metadata)

            probs_ref = F.softmax(logits_ref, dim=1)
            probs_unl = F.softmax(logits_unl, dim=1)

            mse = torch.mean((probs_ref - probs_unl) ** 2, dim=1)
            mse_list.extend(mse.cpu().numpy())

    raw_mse = sum(mse_list) / len(mse_list)
    max_mse = 1.0 / num_classes

    return raw_mse / max_mse

# Utility (AUC)

def compute_auc(model, dataloader, device):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for image_1, image_2, metadata, labels in dataloader:
            image_1 = image_1.to(device)
            image_2 = image_2.to(device)
            metadata = metadata.to(device)

            logits = model(image_1, image_2, metadata)
            probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Classes that actually appear in the ground-truth labels
    valid_classes = np.unique(all_labels)
    print("AUC computed on classes:", valid_classes)

    # Slice only the columns for valid classes
    probs_subset = all_probs[:, valid_classes]

    # Renormalize so each row sums to 1.0 again (restoring a valid probability distribution)
    probs_subset = probs_subset / probs_subset.sum(axis=1, keepdims=True)

    return roc_auc_score(
        all_labels,
        probs_subset,
        multi_class="ovr",
        average="macro",
        labels=valid_classes
    )


# Utility precision-recall AUC:
# NB: This metric is only comparable across models if the dataloader is fixed
# (i.e. always the validation set). SHOULD NOT be used on forget/retain sets for
# cross-model comparison - present classes may vary, making scores incomparable!

def compute_pr_auc(model, dataloader, device, num_classes=11):
    # NB: This metric is only comparable across models if the dataloader is fixed
    # (i.e. always the validation set). Do NOT use on forget/retain sets for
    # cross-model comparison — present classes may vary, making scores incomparable.
    model.eval()
    all_probs = []
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for image_1, image_2, metadata, labels in dataloader:
            image_1 = image_1.to(device)
            image_2 = image_2.to(device)
            metadata = metadata.to(device)
            logits = model(image_1, image_2, metadata)
            probs = F.softmax(logits, dim=1)
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    all_probs_np = all_probs.numpy()
    all_labels_np = all_labels.numpy()

    y_true_oh = label_binarize(all_labels_np, classes=list(range(num_classes)))
    valid_classes = np.unique(all_labels_np).astype(int)
    print("PR-AUC computed on classes:", valid_classes)

    pr_aucs = []
    counts = []
    for i in valid_classes:
        precision, recall, _ = precision_recall_curve(y_true_oh[:, i], all_probs_np[:, i])
        pr_aucs.append(auc(recall, precision))
        counts.append(y_true_oh[:, i].sum())

    pr_aucs = np.array(pr_aucs)
    counts = np.array(counts)
    macro_pr_auc = np.mean(pr_aucs)
    weighted_pr_auc = np.sum(pr_aucs * (counts / counts.sum()))

    print(f"  Macro PR-AUC:    {macro_pr_auc:.4f}")
    print(f"  Weighted PR-AUC: {weighted_pr_auc:.4f}")

    return macro_pr_auc, weighted_pr_auc, all_logits, all_labels, all_probs
   
   
# Version to work with older models that use only one image and metadata
def compute_pr_auc_image_metadata(model, dataloader, device, num_classes=11):
    # NB: This metric is only comparable across models if the dataloader is fixed
    # (i.e. always the validation set). Do NOT use on forget/retain sets for
    # cross-model comparison — present classes may vary, making scores incomparable.
    model.eval()
    all_probs = []
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for image, metadata, labels in dataloader:
            image = image.to(device)
            metadata = metadata.to(device)
            logits = model(image, metadata)
            probs = F.softmax(logits, dim=1)
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    all_probs_np = all_probs.numpy()
    all_labels_np = all_labels.numpy()

    y_true_oh = label_binarize(all_labels_np, classes=list(range(num_classes)))
    valid_classes = np.unique(all_labels_np).astype(int)
    print("PR-AUC computed on classes:", valid_classes)

    pr_aucs = []
    counts = []
    for i in valid_classes:
        precision, recall, _ = precision_recall_curve(y_true_oh[:, i], all_probs_np[:, i])
        pr_aucs.append(auc(recall, precision))
        counts.append(y_true_oh[:, i].sum())

    pr_aucs = np.array(pr_aucs)
    counts = np.array(counts)
    macro_pr_auc = np.mean(pr_aucs)
    weighted_pr_auc = np.sum(pr_aucs * (counts / counts.sum()))

    print(f"  Macro PR-AUC:    {macro_pr_auc:.4f}")
    print(f"  Weighted PR-AUC: {weighted_pr_auc:.4f}")

    return macro_pr_auc, weighted_pr_auc, all_logits, all_labels, all_probs
########### FAIRNESS METRICS


def equalized_odds_by_group(y_true, y_pred, groups):

    results = []

    unique_groups = np.unique(groups)

    for g in unique_groups:

        mask = groups == g

        yt = y_true[mask]
        yp = y_pred[mask]

        TP = np.sum((yp == 1) & (yt == 1))
        TN = np.sum((yp == 0) & (yt == 0))
        FP = np.sum((yp == 1) & (yt == 0))
        FN = np.sum((yp == 0) & (yt == 1))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan

        results.append({
            "group": g,
            "TPR": TPR,
            "FPR": FPR,
            "support": len(yt)
        })

    df = pd.DataFrame(results)

    return df


def equalized_odds_by_group_multiclass(y_true, y_pred, groups, num_classes):
    results = []
    for g in np.unique(groups):
        mask = groups == g
        yt = y_true[mask]
        yp = y_pred[mask]

        tprs = []
        fprs = []

        for c in range(num_classes):
            TP = np.sum((yp == c) & (yt == c))
            FN = np.sum((yp != c) & (yt == c))
            FP = np.sum((yp == c) & (yt != c))
            TN = np.sum((yp != c) & (yt != c))

            tprs.append(TP / (TP + FN) if (TP + FN) > 0 else np.nan)
            fprs.append(FP / (FP + TN) if (FP + TN) > 0 else np.nan)

        results.append({
            "group": g,
            "TPR": np.nanmean(tprs),
            "FPR": np.nanmean(fprs),
            "support": len(yt)
        })

    return pd.DataFrame(results)


def equalized_odds_gap(df):
    tpr = df["TPR"].values
    fpr = df["FPR"].values

    G = len(tpr)
    total = 0

    for i in range(G):
        for j in range(G):
            if i != j:
                diff = (abs(tpr[i] - tpr[j]) + abs(fpr[i] - fpr[j])) / 2
                total += diff

    gap = total / (G * (G - 1))
    return gap


def equalized_odds_by_group_multiclass_std(y_true, y_pred, y_probs, groups, num_classes):
    results = []
    for g in np.unique(groups):
        mask = groups == g
        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_probs[mask]  # shape: (n_samples_in_group, num_classes)

        tprs = []
        fprs = []
        for c in range(num_classes):
            TP = np.sum((yp == c) & (yt == c))
            FN = np.sum((yp != c) & (yt == c))
            FP = np.sum((yp == c) & (yt != c))
            TN = np.sum((yp != c) & (yt != c))
            tprs.append(TP / (TP + FN) if (TP + FN) > 0 else np.nan)
            fprs.append(FP / (FP + TN) if (FP + TN) > 0 else np.nan)

        # Confidence = max softmax probability per sample
        confidence_per_sample = ypr.max(axis=1)

        results.append({
            "group": g,
            "TPR": np.nanmean(tprs),
            "FPR": np.nanmean(fprs),
            "mean_confidence": np.mean(confidence_per_sample),
            "std_confidence": np.std(confidence_per_sample),
            "support": len(yt)
        })
    return pd.DataFrame(results)


def compute_fairness_std(y_true, y_pred, y_probs, df_test_clinical, age_bins, gender, skin, num_classes=11):
    y_true_n = y_true.cpu().numpy()
    y_pred_n = y_pred.cpu().numpy()
    y_probs_n = y_probs.cpu().numpy()  # shape: (N, num_classes)

    age_groups = age_bins.astype(str)

    # Age — requires dropping NaN rows since age_approx can be missing
    test_age_mask = ~df_test_clinical["age_approx"].isna().values
    y_true_age = y_true_n[test_age_mask]
    y_pred_age = y_pred_n[test_age_mask]
    y_probs_age = y_probs_n[test_age_mask]
    age_groups_filtered = age_groups[test_age_mask]

    age_results = equalized_odds_by_group_multiclass_std(
        y_true_age, y_pred_age, y_probs_age, age_groups_filtered, num_classes
    )
    age_gap = equalized_odds_gap(age_results)

    print("\nAge Equalized Odds")
    print(age_results.to_string(index=False))
    print("\nAge EO Gap:", age_gap)

    # Gender
    gender_results = equalized_odds_by_group_multiclass_std(
        y_true_n, y_pred_n, y_probs_n, gender, num_classes
    )
    gender_gap = equalized_odds_gap(gender_results)

    print("\nGender Equalized Odds")
    print(gender_results.to_string(index=False))
    print("\nGender EO Gap:", gender_gap)

    # Skin tone
    skin_results = equalized_odds_by_group_multiclass_std(
        y_true_n, y_pred_n, y_probs_n, skin, num_classes
    )
    skin_gap = equalized_odds_gap(skin_results)

    print("\nSkin Tone Equalized Odds")
    print(skin_results.to_string(index=False))
    print("\nSkin EO Gap:", skin_gap)

    # Overall fairness
    overall_fairness = generalized_fairness_score(age_gap, gender_gap, skin_gap)
    print("\nGeneralized Fairness Score:", overall_fairness)

    return {
        "age_results": age_results,
        "age_gap": age_gap,
        "gender_results": gender_results,
        "gender_gap": gender_gap,
        "skin_results": skin_results,
        "skin_gap": skin_gap,
        "fairness_score": overall_fairness
    }
    

def equalized_odds_gap_nan_safe_2_cat(df, y_true, y_pred, groups):
    """
    Compute Equalized Odds gap, but safely handle NaNs.
    Logs any group skipped due to NaN TPR/FPR and prints its confusion matrix.

    df: output of equalized_odds_by_group (must contain 'group', 'TPR', 'FPR')
    y_true, y_pred: full arrays for computing confusion matrix
    groups: same grouping array used to compute df
    """

    fpr = df["FPR"].values
    fnr = 1 - df["TPR"].values

    # Identify which groups are valid
    mask = ~np.isnan(fpr) & ~np.isnan(fnr)
    skipped = np.where(~mask)[0]

    for idx in skipped:
        group_name = df["group"].iloc[idx]
        group_mask = groups == group_name
        yt = y_true[group_mask]
        yp = y_pred[group_mask]

        TP = np.sum((yp == 1) & (yt == 1))
        TN = np.sum((yp == 0) & (yt == 0))
        FP = np.sum((yp == 1) & (yt == 0))
        FN = np.sum((yp == 0) & (yt == 1))

        print(f"\nSkipping group '{group_name}' due to NaN TPR/FPR. Confusion matrix:")
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        print(f"Total samples: {len(yt)} (Pos: {TP+FN}, Neg: {TN+FP})")

    # Only use valid groups
    fpr = fpr[mask]
    fnr = fnr[mask]

    G = len(fpr)
    if G <= 1:
        return np.nan  # not enough groups to compute gap

    total = 0
    for i in range(G):
        for j in range(G):
            if i != j:
                diff = (abs(fpr[i] - fpr[j]) + abs(fnr[i] - fnr[j])) / 2
                total += diff

    gap = total / (G * (G - 1))
    return gap


def generalized_fairness_score(age_gap, gender_gap, skin_gap):
    fairness_age = 1 - age_gap
    fairness_gender = 1 - gender_gap
    fairness_skin = 1 - skin_gap

    generalized_score = (fairness_age + fairness_gender + fairness_skin) / 3

    return generalized_score


def compute_fairness(y_true, y_pred, df_test_clinical, age_bins, gender, skin, num_classes=11):
    y_true_n = y_true.cpu().numpy()
    y_pred_n = y_pred.cpu().numpy()

    age_groups = age_bins.astype(str)

    # Age — requires dropping NaN rows since age_approx can be missing
    test_age_mask = ~df_test_clinical["age_approx"].isna().values
    y_true_age = y_true_n[test_age_mask]
    y_pred_age = y_pred_n[test_age_mask]
    age_groups_filtered = age_groups[test_age_mask]

    age_results = equalized_odds_by_group_multiclass(y_true_age, y_pred_age, age_groups_filtered, num_classes)
    age_gap = equalized_odds_gap(age_results)

    print("\nAge Equalized Odds")
    print(age_results)
    print("\nAge EO Gap:", age_gap)

    # Gender
    gender_results = equalized_odds_by_group_multiclass(y_true_n, y_pred_n, gender, num_classes)
    gender_gap = equalized_odds_gap(gender_results)

    print("\nGender Equalized Odds")
    print(gender_results)
    print("\nGender EO Gap:", gender_gap)

    # Skin tone
    skin_results = equalized_odds_by_group_multiclass(y_true_n, y_pred_n, skin, num_classes)
    skin_gap = equalized_odds_gap(skin_results)

    print("\nSkin Tone Equalized Odds")
    print(skin_results)
    print("\nSkin EO Gap:", skin_gap)

    # Overall fairness
    overall_fairness = generalized_fairness_score(age_gap, gender_gap, skin_gap)
    print("\nGeneralized Fairness Score:", overall_fairness)

    return {
        "age_results": age_results,
        "age_gap": age_gap,
        "gender_results": gender_results,
        "gender_gap": gender_gap,
        "skin_results": skin_results,
        "skin_gap": skin_gap,
        "fairness_score": overall_fairness
    }

########### CLASSIFIERS -> first experiments for classification of the clinical dataset using images

# 1. Simple image classifier
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        # This guarantees output = [B, 512, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

