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


############# Multimodal: Image + Metadata

# 1. Multimidal: Image + Metadata using learned feature fusion

class ViTFeatureExtractor(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()

        self.backbone = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1
        )

        in_features = self.backbone.heads.head.in_features

        # Remove original classifier
        self.backbone.heads.head = nn.Identity()

        # 512 representation
        self.feature_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = self.backbone(x)     # [B, 768]
        features = self.feature_head(x)  # [B, 512]
        return features


class MetadataEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)   # [B, 512]
        

class MultimodalModel_concat(nn.Module):
    def __init__(self, metadata_dim, num_classes=11, dropout=0.25):
        super().__init__()

        self.image_model = ViTFeatureExtractor()
        self.meta_model = MetadataEncoder(metadata_dim)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, num_classes)
        )

    def forward(self, images, metadata):
        img_feat = self.image_model(images)       # [B,512]
        meta_feat = self.meta_model(metadata)     # [B,512]

        fused = torch.cat([img_feat, meta_feat], dim=1)  # fusion by concat [B, 1024]
        out = self.classifier(fused)
        
        return out
        
        
class MultimodalModel(nn.Module):
    def __init__(self, metadata_dim, num_classes=11, dropout=0.25):
        super().__init__()

        self.image_model = ViTFeatureExtractor()
        self.meta_model = MetadataEncoder(metadata_dim)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, num_classes)
        )

    def forward(self, images, metadata):
        img_feat = self.image_model(images)       # [B,512]
        meta_feat = self.meta_model(metadata)     # [B,512]

        fused = img_feat * meta_feat              # fusion

        out = self.classifier(fused)
        return out


# 2. Multimodal: Image + Metadata but using CrossAttention:

class CrossAttentionModel(nn.Module):
    def __init__(self, metadata_dim, embed_dim=512, num_heads=8, num_classes=1, dropout=0.25):
        super().__init__()
        
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # Drop the classifier head
        vit.heads.head = nn.Identity()
        
        self.vit = vit
        self.image_proj = nn.Linear(768, embed_dim)  
        self.meta_proj = nn.Linear(metadata_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Deep head like the one in the original ViT 1 architecture I tried
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, image, metadata):
        # ViT
        x = self.vit._process_input(image)
        B = x.shape[0]

        cls_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        image_tokens = self.vit.encoder.layers(x)   # [B, N, 768]
        image_tokens = self.image_proj(image_tokens)  # [B, N, 512]

        # Metadata
        meta_tokens = self.meta_proj(metadata).unsqueeze(1)  # [B, 1, 512]

        # Cross-attention
        # Query, Key, Value
        fused, _ = self.cross_attn(meta_tokens, image_tokens, image_tokens)

        # Head
        return self.head(fused.squeeze(1))
        
# Multimodal: Image + Metadata - training, evaluation, freeze/unfreeze layers 

def train_one_epoch_multimodal(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, metadata, labels in loader:
        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, metadata)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_multimodal(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []
    for images, metadata, labels in loader:
        images = images.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)
        
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        
        preds.append(outputs.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    acc = accuracy_score(targets, preds)
    return total_loss / len(loader.dataset), acc


def freeze_backbone_multimodal(model):
    # Freeze the ViT backbone (everything except the head)
    for name, param in model.image_model.backbone.named_parameters():
        if "heads.head" not in name:  # keep ViT head trainable
            param.requires_grad = False

    # Metadata encoder should train
    for param in model.meta_model.parameters():
        param.requires_grad = True

    # Fusion/classifier should train
    for param in model.classifier.parameters():
        param.requires_grad = True
        
        
# Stage 2: Train full model (unfreeze everything)
def unfreeze_all_multimodal(model):
    for param in model.parameters():
        param.requires_grad = True
  
  
def train_model_multimodal(model, train_loader, val_loader, criterion, optimizer, device,
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
        train_loss = train_one_epoch_multimodal(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_multimodal(model, val_loader, criterion, device)
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

    if patience < epochs:
        model.load_state_dict(best_weights)

    else:
        final_checkpoint = {
            "epoch": epoch,
            "model_state": copy.deepcopy(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "history": copy.deepcopy(history),
            "final_val_loss": val_loss
        }

        if save_path is not None:
            if CHECKPOINT_DIR not in save_path:
                if '/' not in save_path:
                    save_path = CHECKPOINT_DIR + '/' + save_path
                else:
                    save_path = CHECKPOINT_DIR + save_path

        torch.save(final_checkpoint, save_path)
        print("Final epoch model saved")

    return model, history
    

        
################### Multimodal: Image + Image

# 1. Multimodal: Image + Image using Mutual CrossAttention

class MutualCrossAttentionModel_images(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_classes=11, dropout=0.25):
        super().__init__()
        
        vit_1 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        vit_2 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Drop the classifier head
        vit_1.heads.head = nn.Identity()
        vit_2.heads.head = nn.Identity()

        
        self.vit_1 = vit_1
        self.vit_2 = vit_2

        self.image_proj_1 = nn.Linear(768, embed_dim)  
        self.image_proj_2 = nn.Linear(768, embed_dim)  

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Deep head like the one in the original ViT 1 architecture I tried
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, image_1, image_2):
        # ViT 1
        x = self.vit_1._process_input(image_1)
        B = x.shape[0]

        cls_token = self.vit_1.class_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.vit_1.encoder.pos_embedding
        x = self.vit_1.encoder.dropout(x)

        image_tokens_1 = self.vit_1.encoder.layers(x)   # [B, N, 768]
        image_tokens_1 = self.image_proj_1(image_tokens_1)  # [B, N, 512]

        # ViT 2 
        x = self.vit_2._process_input(image_2)
        B = x.shape[0]

        cls_token = self.vit_2.class_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.vit_2.encoder.pos_embedding
        x = self.vit_2.encoder.dropout(x)

        image_tokens_2 = self.vit_2.encoder.layers(x)   # [B, N, 768]
        image_tokens_2 = self.image_proj_2(image_tokens_2)  # [B, N, 512]

        # Cross-attention
        fused_1, _ = self.cross_attn(image_tokens_1, image_tokens_2, image_tokens_2)
        fused_2, _ = self.cross_attn(image_tokens_2, image_tokens_1, image_tokens_1)

        x1 = fused_1[:, 0, :]
        x2 = fused_2[:, 0, :]

        x = x1 + x2   # or concat
        # Head

        return self.head(x)

# Multimodal: Image + Image - training, evaluation, freeze/unfreeze layers 

def freeze_backbone_images(model):
    # Freeze both ViT backbones
    for param in model.vit_1.parameters():
        param.requires_grad = False

    for param in model.vit_2.parameters():
        param.requires_grad = False

    # Train projections + attention + head
    for param in model.image_proj_1.parameters():
        param.requires_grad = True

    for param in model.image_proj_2.parameters():
        param.requires_grad = True

    for param in model.cross_attn.parameters():
        param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True
        

def unfreeze_all_images(model):
    for param in model.parameters():
        param.requires_grad = True
        

def train_model_images(model, train_loader, val_loader, criterion, optimizer, device,
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
        train_loss = train_one_epoch_images(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_images(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

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
    
    
@torch.no_grad()
def evaluate_images(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []

    for image_1, image_2, labels in loader:
        image_1 = image_1.to(device)
        image_2 = image_2.to(device)
        labels = labels.to(device)

        outputs = model(image_1, image_2)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * image_1.size(0)

        preds.append(outputs.argmax(dim=1).cpu())
        targets.append(labels.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    acc = accuracy_score(targets, preds)

    return total_loss / len(loader.dataset), acc
    
    
def train_one_epoch_images(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for image_1, image_2, labels in loader:
        image_1 = image_1.to(device)
        image_2 = image_2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(image_1, image_2)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * image_1.size(0)

    return total_loss / len(loader.dataset)
    
    
###################### Mulitmodal: Image + Image + Metadata

# 1. Mutual crossAttention for 3 modalities:

class MutualCrossAttentionModel_3modalities(nn.Module):
    def __init__(
        self,
        metadata_dim,
        embed_dim=512,
        num_heads=8,
        num_classes=11,
        dropout=0.25
    ):
        super().__init__()

        # ViT backbones for both images
        vit_1 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        vit_2 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        vit_1.heads.head = nn.Identity()
        vit_2.heads.head = nn.Identity()

        self.vit_1 = vit_1
        self.vit_2 = vit_2

        # Projections for images and metadata
        self.image_proj_1 = nn.Linear(768, embed_dim)
        self.image_proj_2 = nn.Linear(768, embed_dim)
        self.meta_proj = nn.Linear(metadata_dim, embed_dim)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512), # 256 (would train faster)
            nn.BatchNorm1d(512),  
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, image_1, image_2, metadata):
        
        # IMAGE 1 
        x = self.vit_1._process_input(image_1)
        B = x.shape[0]

        cls_token = self.vit_1.class_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.vit_1.encoder.pos_embedding
        x = self.vit_1.encoder.dropout(x)

        image_tokens_1 = self.vit_1.encoder.layers(x) # [B, N, 768]
        image_tokens_1 = self.image_proj_1(image_tokens_1) # [B, N, D]

        # IMAGE 2 
        x = self.vit_2._process_input(image_2)

        cls_token = self.vit_2.class_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.vit_2.encoder.pos_embedding
        x = self.vit_2.encoder.dropout(x)

        image_tokens_2 = self.vit_2.encoder.layers(x)  # [B, N, 768]
        image_tokens_2 = self.image_proj_2(image_tokens_2)  # [B, N, D]

        # MUTUAL IMAGE FUSION 
        fused_1, _ = self.cross_attn(
            query=image_tokens_1,
            key=image_tokens_2,
            value=image_tokens_2
        )

        fused_2, _ = self.cross_attn(
            query=image_tokens_2,
            key=image_tokens_1,
            value=image_tokens_1
        )

        # Take CLS token
        x1 = fused_1[:, 0, :] # [B, D]
        x2 = fused_2[:, 0, :] # [B, D]

        x_img = x1 + x2 # [B, D]
        
        # METADATA FUSION
        meta_token = self.meta_proj(metadata) # [B, D]
        meta_token = meta_token.unsqueeze(1) # [B, 1, D]

        x_img = x_img.unsqueeze(1) # [B, 1, D]

        fused_meta, _ = self.cross_attn(
            query=meta_token,
            key=x_img,
            value=x_img
        )

        x = fused_meta.squeeze(1) # [B, D]

        # HEAD 
        return self.head(x)
        
# Mulitmodal: Image + Image + Metadata - train, freeze/unfreeze, evaluate:

def freeze_backbone_3_modalities(model):
    # Freeze both ViTs 
    for param in model.vit_1.parameters():
        param.requires_grad = False

    for param in model.vit_2.parameters():
        param.requires_grad = False

    # Train projections 
    for param in model.image_proj_1.parameters():
        param.requires_grad = True

    for param in model.image_proj_2.parameters():
        param.requires_grad = True

    for param in model.meta_proj.parameters():
        param.requires_grad = True

    # Train attention 
    for param in model.cross_attn.parameters():
        param.requires_grad = True

    # Train head 
    for param in model.head.parameters():
        param.requires_grad = True
        
        
def unfreeze_all_3_modalities(model):
    for param in model.parameters():
        param.requires_grad = True
        
        
def train_model_3_modalities(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=50,
    patience=10,
    save_path="best_model.pt"
):

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_3_modalities(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate_3_modalities(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

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

    if epochs > patience:
        model.load_state_dict(best_weights)
    
    return model, history
    

@torch.no_grad()
def evaluate_3_modalities(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []

    for image_1, image_2, metadata, labels in loader:
        image_1 = image_1.to(device)
        image_2 = image_2.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)

        outputs = model(image_1, image_2, metadata)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * image_1.size(0)

        preds.append(outputs.argmax(dim=1).cpu())
        targets.append(labels.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    acc = accuracy_score(targets, preds)

    return total_loss / len(loader.dataset), acc
    
    
def train_one_epoch_3_modalities(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    batch_idx = 0

    GRAD_DEBUG = True
    GRAD_DEBUG_EVERY = 100

    for image_1, image_2, metadata, labels in loader:
        image_1 = image_1.to(device)
        image_2 = image_2.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(image_1, image_2, metadata)

        loss = criterion(outputs, labels)
        loss.backward()

        # For debug why is metadata ignored:
        if GRAD_DEBUG and batch_idx % GRAD_DEBUG_EVERY == 0:

            meta_grad = (
                model.meta_proj.weight.grad
                .abs()
                .mean()
                .item()
            )

            img_grad = (
                model.image_proj_1.weight.grad
                .abs()
                .mean()
                .item()
            )

        optimizer.step()
        batch_idx += 1

        total_loss += loss.item() * image_1.size(0)

    return total_loss / len(loader.dataset)
    
## Enhancement for calculating the forgetness and utility metrics per each epoch and train a fixed number of epochs:
def evaluate_and_collect(model, dataloader, criterion, device, num_classes=11):
    """
    Single forward pass over a dataloader that returns:
    - val_loss, val_acc (from criterion)
    - all_probs, all_labels (for PR-AUC computation outside)
    Replaces separate calls to evaluate_3_modalities + compute_pr_auc.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for image_1, image_2, metadata, labels in dataloader:
            image_1, image_2, metadata, labels = (
                image_1.to(device), image_2.to(device),
                metadata.to(device), labels.to(device)
            )
            logits = model(image_1, image_2, metadata)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    val_loss = total_loss / total
    val_acc = correct / total
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return val_loss, val_acc, all_probs, all_labels


def precompute_reference_probs(reference_model, forget_loader, device):
    """
    Run once before unlearning training starts.
    Returns softmax probabilities for the forget set from the reference model.
    Shape: (N, num_classes) numpy array.
    """
    reference_model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in forget_loader:
            image_1, image_2, metadata, labels = [x.to(device) for x in batch]
            logits = reference_model(image_1, image_2, metadata)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())

    return torch.cat(all_probs).numpy()  # store as numpy, outside GPU


def probs_to_pr_auc(all_probs, all_labels, num_classes=11):
    """
    Compute PR-AUC from already-collected probs and labels.
    No forward pass needed.
    """
    y_true_oh = label_binarize(all_labels, classes=list(range(num_classes)))
    pr_aucs = []
    counts = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, i], all_probs[:, i])
        pr_aucs.append(auc(recall, precision))
        counts.append(y_true_oh[:, i].sum())

    pr_aucs = np.array(pr_aucs)
    counts = np.array(counts)
    macro_pr_auc = np.mean(pr_aucs)
    weighted_pr_auc = np.sum(pr_aucs * (counts / counts.sum()))
    return macro_pr_auc, weighted_pr_auc


def compute_forgetting_mse_from_ref(ref_probs, unlearned_model, forget_loader, device, num_classes=11):
    """Only runs if forget_loader is provided ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â  single pass over forget set."""
    unlearned_model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in forget_loader:
            image_1, image_2, metadata, labels = [x.to(device) for x in batch]
            logits = unlearned_model(image_1, image_2, metadata)
            all_probs.append(F.softmax(logits, dim=1).cpu())

    unl_probs = torch.cat(all_probs).numpy()
    raw_mse = float(np.mean(np.mean((ref_probs - unl_probs) ** 2, axis=1)))
    return raw_mse / (1.0 / num_classes)


def train_model_with_unlearning_metrics(
    model,
    train_loader,
    val_loader,
    forget_loader,
    ref_probs,
    criterion,
    optimizer,
    device,
    save_epoch=3,
    stat_epochs=10,
    num_classes=11,
    save_path=None
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "forgetting_mse": [],
        "val_pr_auc_macro": [],
        "val_pr_auc_weighted": []
    }

    for epoch in range(1, stat_epochs + 1):

        train_loss = train_one_epoch_3_modalities(
            model, train_loader, optimizer, criterion, device
        )

        # Single forward pass for loss + acc + probs
        val_loss, val_acc, val_probs, val_labels = evaluate_and_collect(
            model, val_loader, criterion, device, num_classes=num_classes
        )

        # PR-AUC from already-collected probs no extra forward pass
        macro_pr_auc, weighted_pr_auc = probs_to_pr_auc(val_probs, val_labels, num_classes)

        # One forget set forward pass
        forgetting_mse = compute_forgetting_mse_from_ref(
            ref_probs, model, forget_loader, device, num_classes=num_classes
        )

        print(
            f"Epoch [{epoch}/{stat_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Forgetting MSE: {forgetting_mse:.6f} | "
            f"PR-AUC (macro): {macro_pr_auc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["forgetting_mse"].append(forgetting_mse)
        history["val_pr_auc_macro"].append(macro_pr_auc)
        history["val_pr_auc_weighted"].append(weighted_pr_auc)

        if epoch == save_epoch and save_path is not None:
            checkpoint = {
                "epoch": epoch,
                "model_state": copy.deepcopy(model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "history": copy.deepcopy(history),
            }
            full_path = save_path if '/' in save_path else CHECKPOINT_DIR + '/' + save_path
            torch.save(checkpoint, full_path)
            print(f"  Checkpoint saved at epoch {epoch}")

    return model, history
    
##### NegGrad+ 3 modalities unlearning

def neggrad_plus(
    model,
    forget_loader,
    retain_loader,
    criterion,
    device,
    lr=1e-4,
    steps=500,
    clip_norm=1.0,
    acc_check_every=100,
    acc_threshold=0.1,
    save_path=None
):
    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)

    def get_accuracy(m, loader):
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for image_1, image_2, metadata, labels in loader:
                image_1, image_2, metadata, labels = (
                    image_1.to(device), image_2.to(device),
                    metadata.to(device), labels.to(device)
                )
                logits = m(image_1, image_2, metadata)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        m.train()
        return correct / total if total > 0 else 0.0

    acc_f = get_accuracy(model, forget_loader)
    print(f"  Initial forget accuracy: {acc_f:.4f}")

    for step in range(1, steps + 1):

        # Forget: gradient ascent (manual step, separate from optimizer)
        optimizer.zero_grad()

        try:
            batch_f = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch_f = next(forget_iter)

        if acc_f > acc_threshold:
            image_1, image_2, metadata, labels = [x.to(device) for x in batch_f]
            logits = model(image_1, image_2, metadata)
            loss_f = criterion(logits, labels)
            loss_f.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            # Debug: check gradients are actually non-zero
            total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            if step <= 3:
                print(f"    Step {step} | grad norm before ascent: {total_grad_norm:.6f}")

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data += lr * p.grad  # manual ascent step

        # Retain: gradient descent (via optimizer)
        optimizer.zero_grad()

        try:
            batch_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            batch_r = next(retain_iter)

        image_1, image_2, metadata, labels = [x.to(device) for x in batch_r]
        logits = model(image_1, image_2, metadata)
        loss_r = criterion(logits, labels)
        loss_r.backward()
        optimizer.step()

        if step % acc_check_every == 0:
            acc_f = get_accuracy(model, forget_loader)
            print(f"    Step {step}/500 | Forget acc: {acc_f:.4f}")

    if save_path is not None:
        full_path = save_path if '/' in save_path else CHECKPOINT_DIR + '/' + save_path
        torch.save({
            "model_state": model.state_dict(),
            "lr": lr,
            "steps": steps
        }, full_path)
        print(f"  Saved to {full_path}")

    return model
    
def neggrad_plus_adam(
    model,
    forget_loader,
    retain_loader,
    criterion,
    device,
    lr=1e-4,
    steps=500,
    clip_norm=1.0,
    acc_check_every=100,
    acc_threshold=0.1,
    save_path=None
):
    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)

    def get_accuracy(m, loader):
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for image_1, image_2, metadata, labels in loader:
                image_1, image_2, metadata, labels = (
                    image_1.to(device), image_2.to(device),
                    metadata.to(device), labels.to(device)
                )
                logits = m(image_1, image_2, metadata)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        m.train()
        return correct / total if total > 0 else 0.0

    acc_f = get_accuracy(model, forget_loader)
    print(f"  Initial forget accuracy: {acc_f:.4f}")

    for step in range(1, steps + 1):

        # Forget: gradient ascent (manual step, separate from optimizer)
        optimizer.zero_grad()

        try:
            batch_f = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch_f = next(forget_iter)

        if acc_f > acc_threshold:
            image_1, image_2, metadata, labels = [x.to(device) for x in batch_f]
            logits = model(image_1, image_2, metadata)
            loss_f = criterion(logits, labels)
            loss_f.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            # Debug: check gradients are actually non-zero
            total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            if step <= 3:
                print(f"    Step {step} | grad norm before ascent: {total_grad_norm:.6f}")

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data += lr * p.grad  # manual ascent step

        # Retain: gradient descent (via optimizer) 
        optimizer.zero_grad()

        try:
            batch_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            batch_r = next(retain_iter)

        image_1, image_2, metadata, labels = [x.to(device) for x in batch_r]
        logits = model(image_1, image_2, metadata)
        loss_r = criterion(logits, labels)
        loss_r.backward()
        optimizer.step()

        if step % acc_check_every == 0:
            acc_f = get_accuracy(model, forget_loader)
            print(f"    Step {step}/500 | Forget acc: {acc_f:.4f}")

    if save_path is not None:
        full_path = save_path if '/' in save_path else CHECKPOINT_DIR + '/' + save_path
        torch.save({
            "model_state": model.state_dict(),
            "lr": lr,
            "steps": steps
        }, full_path)
        print(f"  Saved to {full_path}")

    return model
    
# Similar to the approach above, however makes sure extra "repair" steps
# use only descent on the retain set.
def neggrad_plus_repair(
    model,
    forget_loader,
    retain_loader,
    criterion,
    device,
    lr=1e-4,
    steps=500,
    repair_steps=250,
    clip_norm=1.0,
    acc_check_every=100,
    acc_threshold=0.1,
    save_path=None
):
    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)

    def get_accuracy(m, loader):
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for image_1, image_2, metadata, labels in loader:
                image_1, image_2, metadata, labels = (
                    image_1.to(device), image_2.to(device),
                    metadata.to(device), labels.to(device)
                )
                logits = m(image_1, image_2, metadata)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        m.train()
        return correct / total if total > 0 else 0.0

    acc_f = get_accuracy(model, forget_loader)
    print(f"  Initial forget accuracy: {acc_f:.4f}")

    # Phase 1: Ascent on forget + Descent on retain (500 steps = 4 epochs)
    print(f"\n  Phase 1: Ascent + Descent ({steps} steps)")
    for step in range(1, steps + 1):

        # Forget: gradient ascent
        optimizer.zero_grad()
        try:
            batch_f = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch_f = next(forget_iter)

        if acc_f > acc_threshold:
            image_1, image_2, metadata, labels = [x.to(device) for x in batch_f]
            logits = model(image_1, image_2, metadata)
            loss_f = criterion(logits, labels)
            loss_f.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            if step <= 3:
                print(f"    Step {step} | grad norm before ascent: {total_grad_norm:.6f}")
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data += lr * p.grad

        # Retain: gradient descent
        optimizer.zero_grad()
        try:
            batch_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            batch_r = next(retain_iter)

        image_1, image_2, metadata, labels = [x.to(device) for x in batch_r]
        logits = model(image_1, image_2, metadata)
        loss_r = criterion(logits, labels)
        loss_r.backward()
        optimizer.step()

        if step % acc_check_every == 0:
            acc_f = get_accuracy(model, forget_loader)
            print(f"    Step {step}/{steps} | Forget acc: {acc_f:.4f}")

    # Phase 2: Descent only on retain (250 steps = 2 epochs)
    print(f"\n  Phase 2: Retain-only Descent ({repair_steps} steps)")
    retain_iter = iter(retain_loader)  # reset iterator for clean epochs

    for step in range(1, repair_steps + 1):
        optimizer.zero_grad()
        try:
            batch_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            batch_r = next(retain_iter)

        image_1, image_2, metadata, labels = [x.to(device) for x in batch_r]
        logits = model(image_1, image_2, metadata)
        loss_r = criterion(logits, labels)
        loss_r.backward()
        optimizer.step()

        if step % acc_check_every == 0:
            print(f"    Repair step {step}/{repair_steps}")

    acc_f_final = get_accuracy(model, forget_loader)
    print(f"\n  Final forget accuracy: {acc_f_final:.4f}")

    if save_path is not None:
        full_path = save_path if '/' in save_path else CHECKPOINT_DIR + '/' + save_path
        torch.save({
            "model_state": model.state_dict(),
            "lr": lr,
            "steps": steps,
            "repair_steps": repair_steps
        }, full_path)
        print(f"  Saved to {full_path}")

    return model
    
    
#### NegGRad+ version 3

import copy
import torch


def neggrad_plus_repair_weighted(
    model,
    forget_loader,
    retain_loader,
    criterion,
    device,
    lr=1e-4,
    forget_weight=3.0,
    steps=500,
    repair_steps=250,
    clip_norm=5.0,
    acc_check_every=100,
    acc_threshold=0.1,
    save_path=None
):
    model.train()

    # AdamW is too RAM heavy...
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=lr,
    #     weight_decay=1e-4
    # )
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9
    )

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)


    # Evaluation helper

    def evaluate(loader):

        model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():

            for image_1, image_2, metadata, labels in loader:

                image_1 = image_1.to(device)
                image_2 = image_2.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device)

                logits = model(image_1, image_2, metadata)

                loss = criterion(logits, labels)

                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                total_loss += loss.item() * labels.size(0)

        model.train()

        acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return acc, avg_loss

    # Initial evaluation

    acc_f, loss_f = evaluate(forget_loader)

    print(f"Initial forget accuracy: {acc_f:.4f}")
    print(f"Initial forget loss:     {loss_f:.4f}")


    # Phase 1

    print(f"\nPhase 1: Forget ascent + retain descent ({steps} steps)")

    for step in range(1, steps + 1):

        # Get forget batch

        try:
            batch_f = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch_f = next(forget_iter)

        image_1_f, image_2_f, metadata_f, labels_f = [
            x.to(device) for x in batch_f
        ]


        # Get retain batch

        try:
            batch_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            batch_r = next(retain_iter)

        image_1_r, image_2_r, metadata_r, labels_r = [
            x.to(device) for x in batch_r
        ]

        ne

        optimizer.zero_grad(set_to_none=True)

        # Forget pass (gradient ASCENT)

        logits_f = model(image_1_f, image_2_f, metadata_f)

        loss_f_batch = criterion(logits_f, labels_f)

        if acc_f > acc_threshold:

            forget_loss = -forget_weight * loss_f_batch

            forget_loss.backward()

        # Immediately free graph memory
        del logits_f

        # Retain pass (gradient DESCENT)

        logits_r = model(image_1_r, image_2_r, metadata_r)

        loss_r_batch = criterion(logits_r, labels_r)

        loss_r_batch.backward()

        del logits_r

        # Gradient clipping

        if clip_norm is not None:

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_norm
            )
            # Print some debug info
            if step <= 3:
                print(
                    f"Step {step} | "
                    f"grad norm: {grad_norm:.4f} | "
                    f"retain loss: {loss_r_batch.item():.4f} | "
                    f"forget loss: {loss_f_batch.item():.4f}"
                )

        optimizer.step()

        # Monitoring

        if step % acc_check_every == 0:

            acc_f, loss_f = evaluate(forget_loader)

            print(
                f"Step {step}/{steps} | "
                f"Forget acc: {acc_f:.4f} | "
                f"Forget loss: {loss_f:.4f}"
            )

    # Phase 2: repair

    print(f"\nPhase 2: Retain-only repair ({repair_steps} steps)")

    retain_iter = iter(retain_loader)

    for step in range(1, repair_steps + 1):

        try:
            batch_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            batch_r = next(retain_iter)

        image_1, image_2, metadata, labels = [
            x.to(device) for x in batch_r
        ]

        optimizer.zero_grad()

        logits = model(image_1, image_2, metadata)

        loss = criterion(logits, labels)

        loss.backward()

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_norm
            )

        optimizer.step()

        if step % acc_check_every == 0:

            acc_f, loss_f = evaluate(forget_loader)

            print(
                f"Repair step {step}/{repair_steps} | "
                f"Forget acc: {acc_f:.4f}"
            )

    # Final evaluation

    acc_f_final, loss_f_final = evaluate(forget_loader)
    acc_r_final, loss_r_final = evaluate(retain_loader)

    print("\nFinal Results")
    print(f"Forget acc:  {acc_f_final:.4f}")
    print(f"Forget loss: {loss_f_final:.4f}")
    print(f"Retain acc:  {acc_r_final:.4f}")
    print(f"Retain loss: {loss_r_final:.4f}")

    # Save
    if save_path is not None:

        full_path = (
            save_path
            if "/" in save_path
            else CHECKPOINT_DIR + "/" + save_path
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "lr": lr,
                "forget_weight": forget_weight,
                "steps": steps,
                "repair_steps": repair_steps,
            },
            full_path
        )

        print(f"Saved to {full_path}")

    return model