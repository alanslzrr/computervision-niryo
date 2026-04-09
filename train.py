"""
Entrenamiento de CNN para clasificacion de caras de dados de poker.

Carga imagenes desde dataset/, entrena una CNN de 3 bloques conv con PyTorch,
evalua con metricas estandar y exporta el modelo a ONNX para inferencia
con classifier.py (DiceClassifier).

IMPORTANTE: Las imagenes se cargan con cv2 (BGR) y se normalizan a [0,1] sin
convertir a RGB. Esto es critico para mantener paridad con DiceClassifier.preprocess_crop().

Uso:
    python train.py
"""

import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import CNN_INPUT_SIZE, DATASET_DIR, DICE_FACES, ONNX_MODEL_PATH

# ============================================================================
# HIPERPARAMETROS
# ============================================================================

SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT = 0.8
NUM_CLASSES = len(DICE_FACES)
CHECKPOINT_PATH = "best_model.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# DATASET
# ============================================================================

class DiceDataset(Dataset):
    """Dataset que carga crops de dados con OpenCV (BGR) para mantener
    paridad con DiceClassifier.preprocess_crop() en inferencia."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # BGR uint8 — mismo que classifier.py usa en inferencia
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, CNN_INPUT_SIZE)       # (64, 64, 3)
        img = img.astype(np.float32) / 255.0        # [0, 1]
        img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC -> CHW (3, 64, 64)

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


def load_dataset():
    """Carga rutas de imagenes y labels desde DATASET_DIR/{clase}/."""
    paths = []
    labels = []

    for class_idx, face in enumerate(DICE_FACES):
        class_dir = os.path.join(DATASET_DIR, face)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".png"):
                paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)

    return paths, np.array(labels)


# ============================================================================
# MODELO
# ============================================================================

class DiceCNN(nn.Module):
    """CNN de 3 bloques convolucionales segun la arquitectura del proyecto.

    Input:  (B, 3, 64, 64)
    Output: (B, 6) — logits sin softmax
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Block 1: 3x64x64 -> 32x32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Block 2: 32x32x32 -> 64x16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Block 3: 64x16x16 -> 128x8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Clasificador: 128*8*8 = 8192 -> 256 -> num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def make_weighted_sampler(labels):
    """Sampler ponderado para compensar desbalance entre clases."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(labels),
        replacement=True,
    )


# ============================================================================
# ONNX
# ============================================================================

def export_onnx(model, device):
    """Exporta el modelo a ONNX. Output = logits (sin softmax)."""
    model.eval()
    dummy = torch.randn(1, 3, CNN_INPUT_SIZE[0], CNN_INPUT_SIZE[1]).to(device)
    torch.onnx.export(
        model,
        dummy,
        ONNX_MODEL_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )
    print(f"\nModelo exportado a {ONNX_MODEL_PATH}")


def validate_onnx():
    """Verifica que el ONNX exportado carga y produce output correcto."""
    import onnxruntime as ort

    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, CNN_INPUT_SIZE[0], CNN_INPUT_SIZE[1]).astype(np.float32)
    output = session.run(None, {input_name: dummy})[0]
    print(f"ONNX validacion: input {dummy.shape} -> output {output.shape} (esperado: (1, {NUM_CLASSES}))")


# ============================================================================
# MAIN
# ============================================================================

def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) Cargar dataset
    paths, labels = load_dataset()
    if len(paths) == 0:
        print(f"[ERROR] No se encontraron imagenes en {DATASET_DIR}")
        return

    print(f"\nDataset: {len(paths)} imagenes, {NUM_CLASSES} clases")
    for i, face in enumerate(DICE_FACES):
        count = int((labels == i).sum())
        print(f"  {face}: {count}")

    # 2) Split train/val estratificado
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=1 - TRAIN_SPLIT, stratify=labels, random_state=SEED
    )
    print(f"\nTrain: {len(train_paths)} | Val: {len(val_paths)}")

    # 3) Augmentation (solo train)
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    train_dataset = DiceDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DiceDataset(val_paths, val_labels, transform=None)

    # 4) DataLoaders
    train_sampler = make_weighted_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5) Modelo, loss, optimizer
    model = DiceCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: DiceCNN ({total_params:,} parametros)")

    # 6) Training loop
    best_val_acc = 0.0
    print(f"\nEntrenando {NUM_EPOCHS} epochs...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} {'Train Acc':>10} | {'Val Loss':>10} {'Val Acc':>10} | {'Best':>5}")
    print("-" * 72)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        marker = "*" if is_best else ""
        print(
            f"{epoch:>6} | {train_loss:>10.4f} {train_acc:>9.1%} | "
            f"{val_loss:>10.4f} {val_acc:>9.1%} | {marker:>5}"
        )

    # 7) Evaluacion final con mejor checkpoint
    print(f"\nCargando mejor checkpoint (val acc = {best_val_acc:.1%})...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))

    _, final_acc, all_preds, all_labels = evaluate(model, val_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"EVALUACION FINAL — Val Accuracy: {final_acc:.1%}")
    print(f"{'='*50}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    header = "       " + "  ".join(f"{f:>4}" for f in DICE_FACES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>4}" for v in row)
        print(f"  {DICE_FACES[i]:>4}  {row_str}")

    # Per-class metrics
    print(f"\n{classification_report(all_labels, all_preds, target_names=DICE_FACES, zero_division=0)}")

    # 8) Exportar ONNX
    export_onnx(model, device)
    validate_onnx()

    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    train()
