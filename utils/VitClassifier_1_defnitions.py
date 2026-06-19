import torch
import torch.nn as nn
from torchvision import models
import copy
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import numpy as np
import copy
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize


# To make sure that models are saved on the drive, and not only in the memory of Colab.
CHECKPOINT_DIR = '/content/drive/MyDrive/datasets/Master/checkpoints'


############## Unimodal: Image 

# ViT winner (from experiments on 11 classes):
# Vit16 backbone, custom classification head: 512->512->128->11 

# The following custom classification head is also used in the rest of architectures for different modalities.
# nn.Sequential(
        #     nn.Linear(embed_dim, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),

        #     nn.Linear(512, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),

        #     nn.Linear(128, num_classes)
        # )

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=11, dropout=0.5):
        super().__init__()
        # Load pretrained ViT base model
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Replace classification head
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Unimodal Image: evaluation, train, freeze/unfreeze methods:

# Freeze / Unfreeze helpers
def freeze_backbone(model):
    for name, param in model.backbone.named_parameters():
        # Only the final head is trainable
        if "heads.head" not in name:
            param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True

# Training / Evaluation
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds.append(outputs.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    acc = accuracy_score(targets, preds)
    return total_loss / len(loader.dataset), acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                epochs=50, patience=10, save_path="best_model.pt"):
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        # store values
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_checkpoint = {
                "epoch": epoch,
                "model_state": copy.deepcopy(model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "history": copy.deepcopy(history),
                "best_val_loss": best_val_loss
            }
            
            if save_path is not None:
                if CHECKPOINT_DIR not in save_path:
                  if '/' not in save_path:
                    save_path = CHECKPOINT_DIR + '/' + save_path
                  else:
                    save_path = CHECKPOINT_DIR + save_path
                torch.save(best_checkpoint, save_path)
                print("Best model + history saved")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_weights)
    return model, history
    