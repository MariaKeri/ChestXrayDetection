!pip -q install -U timm tqdm scikit-learn

import os
import json
import time
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split

import timm
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


#seed nech su vysledky opakovatelne
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Použité zariadenie:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


#parametre

@dataclass
class Config:
    # Cesty
    dataset: str = "covid" 

    out_dir: str = ""

    # data
    img_size: int = 299
    batch_size: int = 32
    num_workers: int = 2

    # split (70/15/15)
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # model
    model_name: str = "inception_v3"
    num_classes: int = 2
    dropout_head: float = 0.3

    # Optimalizacia
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-4

    # epoch
    epochs_stage1: int = 3
    epochs_stage2: int = 12

    # early stopping
    early_patience: int = 3
    early_min_delta: float = 0.001

    # scheduler
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5

    # Grad-CAM
    gradcam_samples: int = 16  #
    gradcam_alpha: float = 0.35

cfg = Config()
cfg.out_dir = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
OUT = Path(cfg.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

print("Výstupný priečinok:", OUT)

# nacitanie dat a split

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
POS_CLASS = 1

KERNY_ROOT = Path("/content/data/chest_xray")
COVID_ROOT = Path("/content/data/COVID-19_Radiography_Dataset")

def list_images_kermany(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    paths, labels = [], []
    for split in ["train", "val", "test"]:
        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
            d = root / split / cls
            if not d.exists():
                continue
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    if "__MACOSX" in str(p):
                        continue
                    paths.append(p)
                    labels.append(label)
    return np.array(paths), np.array(labels)

def list_images_covid_radiography(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    paths, labels = [], []
    mapping = {
        "Normal": 0,
        "COVID": 1,
        "Viral Pneumonia": 1,
        "Lung_Opacity": 1,
    }
    for folder, y in mapping.items():
        img_dir = root / folder / "images"
        if not img_dir.exists():
            continue
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                paths.append(p)
                labels.append(y)
    return np.array(paths), np.array(labels)

def make_split_stratified(paths: np.ndarray, labels: np.ndarray, seed: int,
                          train_ratio: float, val_ratio: float):

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        paths, labels,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.5,
        random_state=seed,
        stratify=y_tmp
    )
    return {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

#načítanie podľa cfg.dataset
if cfg.dataset.lower() == "kermany":
    paths, labels = list_images_kermany(KERNY_ROOT)
    dataset_name = "KERNY"
elif cfg.dataset.lower() == "covid":
    paths, labels = list_images_covid_radiography(COVID_ROOT)
    dataset_name = "COVID_RAD"
else:
    raise ValueError("cfg.dataset musí byť 'kermany' alebo 'covid'")

splits = make_split_stratified(paths, labels, cfg.seed, cfg.train_ratio, cfg.val_ratio)

print(f"Dataset: {dataset_name}")
print(f"Počet obrázkov: {len(paths)}")
print("Train:", len(splits["train"][0]), "Val:", len(splits["val"][0]), "Test:", len(splits["test"][0]))

# OUT priečinok nech obsahuje názov datasetu
cfg.out_dir = f"run_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
OUT = Path(cfg.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
print("Výstupný priečinok:", OUT)

# uloženie splitu
split_rows = []
for split_name in ["train", "val", "test"]:
    sp_paths, sp_labels = splits[split_name]
    for p, y in zip(sp_paths, sp_labels):
        split_rows.append({"path": str(p), "label": int(y), "split": split_name, "dataset": dataset_name})
splits_df = pd.DataFrame(split_rows)
splits_df.to_csv(OUT / "splits.csv", index=False)
print("splits.csv uložený:", OUT / "splits.csv")

# Dataset + transformácie

class XRayDataset(Dataset):
    def __init__(self, paths: np.ndarray, labels: np.ndarray, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        y = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, y, str(self.paths[idx])

# augmentacie + ImageNet normalizácia
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((320, 320)),
    transforms.RandomCrop(cfg.img_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# Val/Test bez augmentácie
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

train_ds = XRayDataset(*splits["train"], transform=train_tf)
val_ds   = XRayDataset(*splits["val"],   transform=eval_tf)
test_ds  = XRayDataset(*splits["test"],  transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

# Inception-v3 + hlava s dropout

print("\n Inicializujem model:", cfg.model_name)

model = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.num_classes)
# vlastna hlava s dropoutom
in_features = model.get_classifier().in_features if hasattr(model.get_classifier(), "in_features") else None
model.reset_classifier(0)  # odstráni pôvodnú hlavu
if in_features is None:
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Linear):
            in_features = m.in_features
            break
assert in_features is not None, "Nepodarilo sa určiť in_features pre hlavu."

head = nn.Sequential(
    nn.Dropout(p=cfg.dropout_head),
    nn.Linear(in_features, cfg.num_classes)
)
model.classifier = head
model.reset_classifier(cfg.num_classes)
model.classifier = head

model = model.to(device)

# Loss + vahy

train_labels = splits["train"][1]
class_counts = np.bincount(train_labels, minlength=2)

# vahy
class_weights = (len(train_labels) / (cfg.num_classes * class_counts)).astype(np.float32)
class_weights_t = torch.tensor(class_weights, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)

print("Počty tried v train:", class_counts.tolist())
print("Váhy tried:", class_weights.tolist())

# Freeze/Unfreeze

def freeze_all(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_classifier(m: nn.Module):
    for name, p in m.named_parameters():
        if "classifier" in name or name.startswith("fc") or "head" in name:
            p.requires_grad = True

def unfreeze_last_blocks_inception(m: nn.Module):
    for name, p in m.named_parameters():
        if any(k in name for k in ["Mixed_7", "Mixed_6e", "Conv2d_7b", "Conv2d_4a"]):
            p.requires_grad = True

#tréning
def run_epoch_train(model, loader, optimizer) -> float:
    model.train()
    losses = []
    for x, y, _ in tqdm(loader, desc="Tréning", leave=False):
        x = x.to(device)
        y = torch.tensor(y, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0

#evaluacia
@torch.no_grad()
def run_eval(model, loader) -> Dict[str, float]:
    model.eval()
    all_probs, all_preds, all_targets = [], [], []

    for x, y, _ in tqdm(loader, desc="Evaluácia", leave=False):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, POS_CLASS].detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_targets.extend(list(y))

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_prob = np.array(all_probs, dtype=float)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    balanced_acc = 0.5 * (sensitivity + specificity)

    return {
        "acc": float(acc),
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_acc": float(balanced_acc),

        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
    }

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -1e9
        self.bad_epochs = 0

    def step(self, current: float) -> bool:
        if current > self.best + self.min_delta:
            self.best = current
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience

# feature extraction

print("\n feature extraction")

freeze_all(model)
unfreeze_classifier(model)

params_stage1 = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params_stage1, lr=cfg.lr_head, weight_decay=cfg.weight_decay)

history = []

for epoch in range(1, cfg.epochs_stage1 + 1):
    train_loss = run_epoch_train(model, train_loader, optimizer)
    val_metrics = run_eval(model, val_loader)
    row = {
        "stage": 1,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_acc": val_metrics["acc"],
        "val_auc": val_metrics["auc"],
        "val_bal_acc": val_metrics["balanced_acc"],
        "val_sens": val_metrics["sensitivity"],
        "val_spec": val_metrics["specificity"],
        "lr": optimizer.param_groups[0]["lr"],
    }
    history.append(row)
    print(f"[S1 E{epoch}/{cfg.epochs_stage1}] loss={train_loss:.4f} | val_auc={val_metrics['auc']:.4f} | val_acc={val_metrics['acc']:.4f}")

# fine-tuning

print("\n Staged fine-tuning")

freeze_all(model)
unfreeze_classifier(model)
unfreeze_last_blocks_inception(model)

params_stage2 = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params_stage2, lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
)

early = EarlyStopping(patience=cfg.early_patience, min_delta=cfg.early_min_delta)

best_path = OUT / "best_model.pt"
best_val_auc = -1e9

for epoch in range(1, cfg.epochs_stage2 + 1):
    train_loss = run_epoch_train(model, train_loader, optimizer)
    val_metrics = run_eval(model, val_loader)

    scheduler.step(val_metrics["auc"])

    if val_metrics["auc"] > best_val_auc:
        best_val_auc = val_metrics["auc"]
        torch.save({
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "best_val_auc": best_val_auc,
        }, best_path)

    row = {
        "stage": 2,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_acc": val_metrics["acc"],
        "val_auc": val_metrics["auc"],
        "val_sens": val_metrics["sensitivity"],
        "val_spec": val_metrics["specificity"],
        "val_bal_acc": val_metrics["balanced_acc"],

        "lr": optimizer.param_groups[0]["lr"],
    }
    history.append(row)

    print(f"[S2 E{epoch}/{cfg.epochs_stage2}] loss={train_loss:.4f} | val_auc={val_metrics['auc']:.4f} | val_acc={val_metrics['acc']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

    # Early stopping
    if early.step(val_metrics["auc"]):
        print(f" Early stopping: val_auc sa nezlepšila {cfg.early_patience} epoch(y).")
        break

print("Best checkpoint uložený:", best_path, "best_val_auc=", best_val_auc)

# Vyhodnotenie na TEST

print("\n BEST model na test sete")

ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

test_metrics = run_eval(model, test_loader)

bal_acc = test_metrics["balanced_acc"]
print(f"TEST Balanced Acc: {bal_acc:.4f}")

acc = test_metrics["acc"]
auc = test_metrics["auc"]
sens = test_metrics["sensitivity"]
spec = test_metrics["specificity"]

print(f"TEST Accuracy:     {acc:.4f}")
print(f"TEST Sensitivity:  {sens:.4f} (PNEUMONIA)")
print(f"TEST Specificity:  {spec:.4f} (NORMAL)")
print(f"TEST AUC-ROC:      {auc:.4f}")

# Uloženie confusion matrix
cm = np.array([[test_metrics["tn"], test_metrics["fp"]],
               [test_metrics["fn"], test_metrics["tp"]]])

plt.figure()
plt.imshow(cm)
plt.xticks([0,1], ["NORMAL","PNEUMONIA"])
plt.yticks([0,1], ["NORMAL","PNEUMONIA"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i,j]), ha="center", va="center")
plt.title("Confusion Matrix (TEST)")
plt.xlabel("Predikcia")
plt.ylabel("Skutočnosť")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix_test.png", dpi=200)
plt.close()

# Uloženie metrik
metrics_out = {
    "test_accuracy": acc,
    "test_sensitivity": sens,
    "test_specificity": spec,
    "test_auc": auc,
    "tn": test_metrics["tn"], "fp": test_metrics["fp"], "fn": test_metrics["fn"], "tp": test_metrics["tp"],
    "test_balanced_accuracy": bal_acc,

    "best_val_auc": float(best_val_auc),
}
with open(OUT / "metrics_test.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

pd.DataFrame([metrics_out]).to_csv(OUT / "metrics_test.csv", index=False)

y_true = test_metrics["y_true"]
y_prob = test_metrics["y_prob"]

# ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (TEST)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "roc_curve_test.png", dpi=200)
plt.close()

# calibration curve + Brier
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
brier = brier_score_loss(y_true, y_prob)

plt.figure()
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0,1],[0,1],"--")
plt.xlabel("Predikovaná pravdepodobnosť")
plt.ylabel("Skutočný podiel pozitívnych")
plt.title(f"Calibration Curve (TEST) | Brier={brier:.4f}")
plt.tight_layout()
plt.savefig(OUT / "calibration_curve_test.png", dpi=200)
plt.close()

# Uloženie brier
metrics_out["test_brier"] = float(brier)
with open(OUT / "metrics_test.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

# Graf tréningu (loss + val AUC)

hist_df = pd.DataFrame(history)
hist_df.to_csv(OUT / "train_history.csv", index=False)

# Loss
plt.figure()
plt.plot(hist_df["train_loss"].values)
plt.title("Train Loss (all epochs)")
plt.xlabel("epoch index")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig(OUT / "train_loss_curve.png", dpi=200)
plt.close()

# Val AUC
plt.figure()
plt.plot(hist_df["val_auc"].values)
plt.title("Validation AUC-ROC (all epochs)")
plt.xlabel("epoch index")
plt.ylabel("AUC")
plt.tight_layout()
plt.savefig(OUT / "val_auc_curve.png", dpi=200)
plt.close()

# Grad-CAM

class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.target_layer = self._find_last_conv_layer()
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _find_last_conv_layer(self) -> nn.Module:
        last = None
        for _, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("Nenašla sa Conv2d vrstva pre Grad-CAM.")
        return last

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def make_cam(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        # Global average pooling
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
        return cam

def tensor_to_img_denorm(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clone()
    mean = torch.tensor([0.485,0.456,0.406])[:, None, None]
    std  = torch.tensor([0.229,0.224,0.225])[:, None, None]
    x = x * std + mean
    x = torch.clamp(x, 0, 1)
    return np.transpose(x.numpy(), (1,2,0))

gradcam = GradCAM(model)

# vzorky na vizualizáciu
y_pred = test_metrics["y_pred"]
paths_test = splits_df[splits_df["split"]=="test"]["path"].values
y_true_arr = y_true

correct_idx = np.where(y_true_arr == y_pred)[0]
wrong_idx   = np.where(y_true_arr != y_pred)[0]

k = cfg.gradcam_samples
sel = []
if len(wrong_idx) > 0:
    sel_wrong = wrong_idx[: min(len(wrong_idx), k//2)]
    sel.extend(sel_wrong.tolist())
sel_correct = correct_idx[: max(0, k - len(sel))]
sel.extend(sel_correct.tolist())
sel = sel[:k]

gallery_dir = OUT / "gradcam_gallery"
gallery_dir.mkdir(parents=True, exist_ok=True)

for rank, idx in enumerate(tqdm(sel, desc="Grad-CAM", leave=False)):
    # načítanie obrázka
    p = Path(paths_test[idx])
    img = Image.open(p).convert("L")
    x = eval_tf(img).unsqueeze(0).to(device)

    # predikcia
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0, POS_CLASS].item()
        pred = int(torch.argmax(logits, dim=1).item())
        true = int(y_true_arr[idx])

    # CAM pre PNEUMONIA
    cam = gradcam.make_cam(x, class_idx=POS_CLASS)

    # obrázok pre overlay
    vis = tensor_to_img_denorm(x[0])
    # cam resize
    cam_img = Image.fromarray((cam*255).astype(np.uint8)).resize((cfg.img_size, cfg.img_size))
    cam_np = np.array(cam_img).astype(np.float32) / 255.0

    # overlay
    overlay = vis.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + cfg.gradcam_alpha * cam_np, 0, 1)

    # uloženie
    title = f"{rank:02d}_true{true}_pred{pred}_pneuProb{prob:.3f}"
    plt.figure(figsize=(8,3))
    plt.subplot(1,3,1); plt.imshow(vis); plt.axis("off"); plt.title("Input")
    plt.subplot(1,3,2); plt.imshow(cam_np, cmap="jet"); plt.axis("off"); plt.title("CAM")
    plt.subplot(1,3,3); plt.imshow(overlay); plt.axis("off"); plt.title("Overlay")
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(gallery_dir / f"{title}.png", dpi=180)
    plt.close()

print("Grad-CAM galéria uložená:", gallery_dir)

# model card

model_card = f"""# Model Card: Inception-v3 – Chest X-Ray Pneumonia

## Task
Binary classification of chest X-ray images: **NORMAL (0)** vs **PNEUMONIA (1)**.

## Dataset
Public dataset: *Chest X-Ray Images (Pneumonia)* (HF mirror: `hf-vision/chest-xray-pneumonia`).
Images were organized into folders and re-split into **70/15/15** (train/val/test).
Split file: `splits.csv`.

## Preprocessing
- Grayscale → RGB (3 channels)
- Train: Resize 320×320 → RandomCrop 299×299, HorizontalFlip, Rotation
- Val/Test: Resize 299×299
- ImageNet normalization

## Model
- Base: `{cfg.model_name}` pretrained on ImageNet (timm)
- Head: Dropout(p={cfg.dropout_head}) + Linear({cfg.num_classes})

## Training procedure
- Stage 1 (feature extraction): backbone frozen, head trained for {cfg.epochs_stage1} epochs
- Stage 2 (staged fine-tuning): last blocks + head unfrozen, trained up to {cfg.epochs_stage2} epochs
- Loss: CrossEntropy with class weights
- Optimizer: Adam, weight decay={cfg.weight_decay}
- LR schedule: ReduceLROnPlateau on validation AUC
- Early stopping: patience={cfg.early_patience}, min_delta={cfg.early_min_delta}
- Best checkpoint selected by **validation AUC-ROC**

## Test results (best checkpoint)
- Accuracy: {acc:.4f}
- Sensitivity (PNEUMONIA): {sens:.4f}
- Specificity (NORMAL): {spec:.4f}
- AUC-ROC: {auc:.4f}
- Brier score (calibration): {brier:.4f}
"""

with open(OUT / "MODEL_CARD.md", "w", encoding="utf-8") as f:
    f.write(model_card)

print("\nMODEL_CARD.md uložený:", OUT / "MODEL_CARD.md")

# Zhrnutie

print("splits.csv:", OUT / "splits.csv")
print("best_model.pt:", best_path)
print("metrics_test.json:", OUT / "metrics_test.json")
print("ROC:", OUT / "roc_curve_test.png")
print("Calibration:", OUT / "calibration_curve_test.png")
print("Confusion matrix:", OUT / "confusion_matrix_test.png")
print("Train curves:", OUT / "train_loss_curve.png", "a", OUT / "val_auc_curve.png")
print("Grad-CAM gallery:", gallery_dir)
print("Model card:", OUT / "MODEL_CARD.md")
