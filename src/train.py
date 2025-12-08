import torch
from torch.utils.data import DataLoader
from typing import Tuple


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch
    return: (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)          # (B, 1, 187)
        y = y.to(device)          # (B,)

        optimizer.zero_grad()
        logits = model(x)         # (B, 5)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += batch_size

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Common evaluation function for validation/testing
    return: (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += batch_size

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc
