import torch
import torch.nn as nn
from torchvision import models
import copy
from sklearn.metrics import accuracy_score

CHECKPOINT_DIR = '/content/drive/MyDrive/datasets/Master/checkpoints'


class ViTClassifierCLSPool(nn.Module):
    """
    Combines CLS token + mean patch pooling
    More robust feature representation
    """
    def __init__(self, num_classes=11, dropout=0.5):
        super().__init__()

        self.vit = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1
        )

        hidden_dim = self.vit.hidden_dim

        # Remove original head
        self.vit.heads = nn.Identity()

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        n = x.shape[0]

        # ViT forward until encoder output
        x = self.vit._process_input(x)
        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.encoder(x)

        # CLS token
        cls_feat = x[:, 0]

        # Mean pooling over patch tokens
        patch_feat = x[:, 1:].mean(dim=1)

        features = torch.cat([cls_feat, patch_feat], dim=1)
        return self.classifier(features)

# Freeze / Unfreeze helpers
def freeze_backbone(model):
    for param in model.vit.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_backbone(model):
    for param in model.parameters():
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

        # store values
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            # if save_path is not None:
            #   torch.save(model.state_dict(), save_path)
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