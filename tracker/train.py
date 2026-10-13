
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,

)

from data import (
    train_loader,
    val_loader,
    test_loader,
    train_dataset,
    val_dataset,
    test_dataset,
)

from model import build_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
NUM_CLASSES = 4

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 10

PHASE1_LR = 1e-3
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

PHASE1_PATIENCE = 3
PHASE2_PATIENCE = 4
MIN_DELTA = 0.001

CHECKPOINT_PATH = Path(__file__).resolve().parent / "best_model.pth"


# ============================================================
# 2. REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. VERIFY DATASET
# ============================================================

class_names = train_dataset.classes

assert len(class_names) == NUM_CLASSES, (
    f"Expected {NUM_CLASSES} classes, found {class_names}"
)

assert train_dataset.class_to_idx == val_dataset.class_to_idx, (
    "Train and validation class mappings differ."
)

assert train_dataset.class_to_idx == test_dataset.class_to_idx, (
    "Train and test class mappings differ."
)

print("\nClass mapping:", train_dataset.class_to_idx)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# 4. BUILD MODEL AND LOSS FUNCTION
# ============================================================

model = build_model(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=True,
).to(device)

criterion = nn.CrossEntropyLoss(
    label_smoothing=LABEL_SMOOTHING
)

print("\nModel loaded successfully.")


# ============================================================
# 5. EARLY STOPPING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0

    def step(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


# ============================================================
# 6. TRAINING FUNCTION
# ============================================================


def train_one_epoch(model, loader, optimizer, phase):
    model.train()

    if phase == 1:
        model.features.eval()
    else:
        model.features[:-2].eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    avg_loss = total_loss / total_samples

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )

    return avg_loss, accuracy, precision, recall, macro_f1



# ============================================================
# 7. VALIDATION FUNCTION
# ============================================================


def validate(model, loader):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    avg_loss = total_loss / total_samples

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        all_labels, all_predictions,
        average="macro", zero_division=0
    )

    return avg_loss, accuracy, precision, recall, macro_f1



# ============================================================
# 8. SAVE THE BEST CHECKPOINT ACROSS BOTH PHASES
# ============================================================

best_val_f1 = -float("inf")


def save_best_checkpoint(model, val_f1, phase, epoch):
    global best_val_f1

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        torch.save(
            {
                "model_state_dict": copy.deepcopy(
                    model.state_dict()
                ),
                "best_val_f1": best_val_f1,
                "phase": phase,
                "epoch": epoch,
                "class_to_idx": train_dataset.class_to_idx,
            },
            CHECKPOINT_PATH,
        )

        print("Best checkpoint updated!")


# ============================================================
# 9. PHASE 1: TRAIN CLASSIFIER ONLY
# ============================================================

print("\n========== PHASE 1: FROZEN BACKBONE ==========")

# Freeze the feature extractor.
for param in model.features.parameters():
    param.requires_grad = False

# Train the new classifier.
for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = Adam(
    model.classifier.parameters(),
    lr=PHASE1_LR,
    weight_decay=WEIGHT_DECAY,
)

early_stopping = EarlyStopping(
    patience=PHASE1_PATIENCE,
    min_delta=MIN_DELTA,
)

for epoch in range(PHASE1_EPOCHS):
    train_loss, train_acc, train_precision, train_recall, train_f1 = (
    train_one_epoch(model, train_loader, optimizer, phase=1)
)

    val_loss, val_acc, val_precision, val_recall, val_f1 = (
    validate(model, val_loader)
    )

    print(
    f"\nPhase 1 | Epoch {epoch + 1}/{PHASE1_EPOCHS}\n"
    f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
    f"Precision: {train_precision:.4f} | "
    f"Recall: {train_recall:.4f} | F1: {train_f1:.4f}\n"
    f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
    f"Precision: {val_precision:.4f} | "
    f"Recall: {val_recall:.4f} | F1: {val_f1:.4f}"
)

    save_best_checkpoint(
        model, val_f1, phase=1, epoch=epoch + 1
    )

    if early_stopping.step(val_f1):
        print("Early stopping triggered in Phase 1.")
        break


# ============================================================
# 10. RESTORE BEST MODEL BEFORE PHASE 2
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])

print(
    "\nBest validation Macro-F1 after Phase 1:",
    checkpoint["best_val_f1"],
)


# ============================================================
# 11. PHASE 2: FINE-TUNE LAST TWO FEATURE BLOCKS
# ============================================================

print("\n========== PHASE 2: FINE-TUNING ==========")

# Freeze every feature block first.
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze the last two blocks.
for param in model.features[-2:].parameters():
    param.requires_grad = True

# Keep the classifier trainable.
for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = Adam(
    [
        {
            "params": model.features[-2:].parameters(),
            "lr": BACKBONE_LR,
        },
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

early_stopping = EarlyStopping(
    patience=PHASE2_PATIENCE,
    min_delta=MIN_DELTA,
)

for epoch in range(PHASE2_EPOCHS):
    train_loss, train_acc, train_precision, train_recall, train_f1 = (
    train_one_epoch(model, train_loader, optimizer, phase=2)
)

    val_loss, val_acc, val_precision, val_recall, val_f1 = (
    validate(model, val_loader)
)

    print(
    f"\nPhase 1 | Epoch {epoch + 1}/{PHASE1_EPOCHS}\n"
    f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
    f"Precision: {train_precision:.4f} | "
    f"Recall: {train_recall:.4f} | F1: {train_f1:.4f}\n"
    f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
    f"Precision: {val_precision:.4f} | "
    f"Recall: {val_recall:.4f} | F1: {val_f1:.4f}"
)

    save_best_checkpoint(model, val_f1, phase=1, epoch=epoch + 1)

    if early_stopping.step(val_f1):
        print("Early stopping triggered in Phase 2.")
        break


# ============================================================
# 12. LOAD BEST CHECKPOINT FROM EITHER PHASE
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\n========== BEST CHECKPOINT ==========")
print(f"Best validation Macro-F1: {checkpoint['best_val_f1']:.4f}")
print(f"Best phase: {checkpoint['phase']}")
print(f"Best epoch in that phase: {checkpoint['epoch']}")


# ============================================================
# 13. FINAL TEST PREDICTIONS
# ============================================================

# The test set is evaluated only after model selection is complete.
all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

y_true = np.asarray(all_labels)
y_pred = np.asarray(all_predictions)
y_prob = np.asarray(all_probabilities)


# ============================================================
# 14. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

precision_macro = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
precision_weighted = precision_score(
    y_true, y_pred, average="weighted", zero_division=0
)

recall_macro = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
recall_weighted = recall_score(
    y_true, y_pred, average="weighted", zero_division=0
)

f1_macro = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)
f1_weighted = f1_score(
    y_true, y_pred, average="weighted", zero_division=0
)

print("\n========== FINAL TEST METRICS ==========")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Macro Precision    : {precision_macro:.4f}")
print(f"Weighted Precision : {precision_weighted:.4f}")
print(f"Macro Recall       : {recall_macro:.4f}")
print(f"Weighted Recall    : {recall_weighted:.4f}")
print(f"Macro F1-score     : {f1_macro:.4f}")
print(f"Weighted F1-score  : {f1_weighted:.4f}")


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )
)


# ============================================================
# 17. PER-CLASS AND OVERALL ROC-AUC
# ============================================================

print("\n========== PER-CLASS ROC-AUC ==========")

for i, class_name in enumerate(class_names):
    binary_labels = (y_true == i).astype(int)

    if len(np.unique(binary_labels)) == 2:
        auc = roc_auc_score(
            binary_labels,
            y_prob[:, i],
        )
        print(f"{class_name}: {auc:.4f}")
    else:
        print(
            f"{class_name}: Undefined "
            "(missing positive or negative examples)"
        )

if len(np.unique(y_true)) == NUM_CLASSES:
    auc_macro = roc_auc_score(
        y_true,
        y_prob,
        labels=list(range(NUM_CLASSES)),
        multi_class="ovr",
        average="macro",
    )

    auc_weighted = roc_auc_score(
        y_true,
        y_prob,
        labels=list(range(NUM_CLASSES)),
        multi_class="ovr",
        average="weighted",
    )

    print(f"\nMacro ROC-AUC (OvR): {auc_macro:.4f}")
    print(f"Weighted ROC-AUC    : {auc_weighted:.4f}")
else:
    print(
        "\nOverall multiclass ROC-AUC unavailable: "
        "at least one class is missing from the test set."
    )


# ============================================================
# 18. SAVE FINAL TEST METRICS SUMMARY
# ============================================================

metrics_path = (
    Path(__file__).resolve().parent / "test_metrics.txt"
)

with open(metrics_path, "w", encoding="utf-8") as file:
    file.write("FINAL TEST METRICS\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro Precision: {precision_macro:.4f}\n")
    file.write(f"Weighted Precision: {precision_weighted:.4f}\n")
    file.write(f"Macro Recall: {recall_macro:.4f}\n")
    file.write(f"Weighted Recall: {recall_weighted:.4f}\n")
    file.write(f"Macro F1-score: {f1_macro:.4f}\n")
    file.write(f"Weighted F1-score: {f1_weighted:.4f}\n")

print(f"\nMetrics saved to: {metrics_path}")
print(f"Best model saved to: {CHECKPOINT_PATH}")
print("\nTraining and evaluation completed.")
