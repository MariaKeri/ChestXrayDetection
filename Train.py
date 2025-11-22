import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Medical Image Classification via Transfer Learning (Inception-v3 on CXR)"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Cesta k priečinku s podpriečinkami train/val/test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Ak je nastavené, zmrazí všetky vrstvy okrem hlavy")
    args = parser.parse_args()
    return args


def get_transforms():
    """
    Inception-v3 očakáva vstupy 299x299, normalizované na ImageNet štatistiky.
    Pre train pridáme augmentácie, pre val/test iba resize + center crop.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((320, 320)),
        transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((320, 320)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, eval_tf


def create_dataloaders(data_dir, batch_size, num_workers):
    data_dir = Path(data_dir)
    train_tf, eval_tf = get_transforms()

    image_datasets = {
        "train": datasets.ImageFolder(data_dir / "train", transform=train_tf),
        "val": datasets.ImageFolder(data_dir / "val", transform=eval_tf),
        "test": datasets.ImageFolder(data_dir / "test", transform=eval_tf),
    }

    dataloaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True
        )
        for split, ds in image_datasets.items()
    }

    class_names = image_datasets["train"].classes
    return image_datasets, dataloaders, class_names


def build_model(num_classes: int, freeze_backbone: bool = False):
    """
    Načítame Inception-v3 predtréningom na ImageNet a nahradíme klasifikačné heady.
    """
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights)

    in_features_main = model.fc.in_features
    model.fc = nn.Linear(in_features_main, num_classes)

    if model.aux_logits:
        in_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "AuxLogits.fc" not in name:
                param.requires_grad = False

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        if isinstance(outputs, tuple):
            main_outputs, aux_outputs = outputs
            loss1 = criterion(main_outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
            preds = main_outputs.argmax(dim=1)
        else:
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    all_labels = []
    all_preds = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  

        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    return epoch_loss, epoch_acc, all_labels, all_preds


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # 1. Data 
    print("Loading data...")
    datasets_dict, dataloaders, class_names = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    num_classes = len(class_names)
    print(f"Classes: {class_names} (num_classes={num_classes})")
    
    train_ds = datasets_dict["train"]
    labels = [label for _, label in train_ds.samples]
    counts = Counter(labels)
    print("Train class counts:", counts)  # len na kontrolu

    class_weights = []
    for cls_idx in range(num_classes):
        class_weights.append(1.0 / counts[cls_idx])

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


    # 2. Model
    print("Building model...")
    model = build_model(num_classes=num_classes, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # 3. Criterion+optimizer
    num_classes = len(class_names)
    class_weights = []
    for cls_idx in range(num_classes):
        class_weights.append(1.0 / counts[cls_idx])

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params_to_update,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_inception_v3.pth")

    # 4. Tg slucka
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, dataloaders["val"], criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Train loss: {train_loss:.4f} | acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | acc: {val_acc:.4f}")
        print(f"Epoch time: {elapsed:.1f} s")

        # naj model podla val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"==> New best model saved to {best_model_path}")

    print(f"\nTraining finished. Best val accuracy: {best_val_acc:.4f}")

    # 5. Nacitame naj model + test
    print("\nEvaluating on test set with best model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, dataloaders["test"], criterion, device
    )
    print(f"Test loss: {test_loss:.4f} | acc: {test_acc:.4f}")

    # 6. report + cfm
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open(os.path.join(args.output_dir, "test_report.txt"), "w") as f:
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(str(cm))

    print("\nClassification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
