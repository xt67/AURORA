import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score, roc_auc_score

from src.classifier.model import MedicalImageClassifier, get_class_weights
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

# PyTorch 2.2 compatible AMP
try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:  # pragma: no cover - for older PyTorch
    GradScaler = None
    autocast = None


def train_classifier(
    config_path: str,
    train_loader,
    val_loader,
    device: torch.device,
    use_synthetic: bool = False,
    synthetic_loader=None,
    output_dir: str = "outputs",
    seed: Optional[int] = None,
    augmentation_mode: str = "conventional",
) -> nn.Module:
    """Train classifier model.
    
    Args:
        config_path: Path to config YAML.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Device to train on.
        use_synthetic: Whether to include synthetic images.
        synthetic_loader: Synthetic image loader.
        output_dir: Output directory for checkpoints and logs.
        seed: Random seed (overrides config if provided). If None, uses config seed.
        augmentation_mode: "none" | "conventional" - controls stochastic augmentations.
    
    Returns:
        Trained model with best validation checkpoint loaded.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Use provided seed or config seed; do not override if explicitly passed
    if seed is not None:
        config["seed"] = seed
    set_seed(config["seed"])
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Training and validation loaders must be nonempty")
    if use_synthetic and (synthetic_loader is None or len(synthetic_loader) == 0):
        raise ValueError("Synthetic augmentation requires a nonempty synthetic loader")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("cls_train", str(output_path / "logs"))
    
    num_classes = config.get("num_classes", 2)
    pretrained = config.get("pretrained", True)
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    epochs = config.get("epochs", 20)
    early_stop_patience = config.get("early_stop_patience", 10)
    use_amp = config.get("use_amp", True) and device.type == "cuda"
    
    model = MedicalImageClassifier(num_classes=num_classes, pretrained=pretrained).to(device)
    
    # Compute class weights from the base dataset (handles Subset)
    class_weights = get_class_weights(train_loader.dataset, num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # AMP gradient scaler (PyTorch 2.2 compatible)
    scaler = GradScaler(enabled=use_amp) if use_amp and GradScaler is not None else None
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = output_path / "models" / "classifier_best.pth"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training classifier for {epochs} epochs")
    logger.info(f"Augmentation mode: {augmentation_mode}")
    logger.info(f"Use synthetic augment: {use_synthetic}")
    logger.info(f"AMP enabled: {use_amp}")
    logger.info(f"Class weights: {class_weights.cpu().numpy() if class_weights is not None else None}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        synthetic_iter = iter(synthetic_loader) if use_synthetic else None
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            if use_synthetic and synthetic_loader is not None:
                try:
                    syn_images, syn_labels = next(synthetic_iter)
                except StopIteration:
                    synthetic_iter = iter(synthetic_loader)
                    syn_images, syn_labels = next(synthetic_iter)
                images = torch.cat([images, syn_images.to(device)], dim=0)
                labels = torch.cat([labels, syn_labels.to(device)], dim=0)
            
            optimizer.zero_grad()
            
            # AMP forward/backward (PyTorch 2.2 compatible)
            if use_amp and autocast is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            if use_amp and autocast is not None:
                with autocast():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        preds = torch.argmax(outputs, dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
            else:
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
        val_precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        
        logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}, Recall={val_recall:.4f}, Prec={val_precision:.4f}, F1={val_f1:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with val_loss={avg_val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Restore best checkpoint before returning
    logger.info(f"Loading best model from {best_model_path} (val_loss={best_val_loss:.4f})")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    logger.info("Training complete!")
    return model


def evaluate_classifier(model: nn.Module, test_loader, device: torch.device, use_amp: bool = True) -> dict:
    """Evaluate classifier on test set.
    
    Args:
        model: Trained classifier model.
        test_loader: Test data loader.
        device: Device to evaluate on.
        use_amp: Whether to use AMP for inference.
    
    Returns:
        Dictionary containing evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        eval_amp = use_amp and device.type == "cuda" and autocast is not None
        if eval_amp:
            with autocast():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())  # TB class probability
        else:
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # TB class probability
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    if len(all_labels) == 0:
        raise ValueError("Evaluation loader must be nonempty")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = (cm.ravel().tolist() + [0, 0, 0, 0])[:4] if cm.size == 4 else (0, 0, 0, 0)
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
    }
    
    # AUPRC (primary complementary metric for imbalanced data)
    if len(np.unique(all_labels)) < 2:
        metrics.update(auprc=None, roc_auc=None)
        return metrics
    try:
        metrics["auprc"] = average_precision_score(all_labels, all_probs)
    except Exception:
        metrics["auprc"] = None
    
    # ROC-AUC
    try:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except Exception:
        metrics["roc_auc"] = None
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--augmentation_mode", type=str, default="conventional", choices=["none", "conventional"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    from src.data_loader import create_dataloaders
    
    # Load config to get data_dir
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    data_dir = config.get("data_dir", args.data_dir)
    
    # Create dataloaders with appropriate settings
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        batch_size=config.get("batch_size", 16),
        image_size=128,
        num_workers=0,  # Windows-safe
        seed=config.get("seed", 42),
        augmentation_mode=args.augmentation_mode,
        for_classifier=True,  # Use ImageNet normalization for pretrained ResNet
    )
    
    model = train_classifier(
        args.config,
        train_loader,
        val_loader,
        device,
        output_dir=args.output_dir,
        seed=args.seed,
        augmentation_mode=args.augmentation_mode,
    )
    
    metrics = evaluate_classifier(model, test_loader, device)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
