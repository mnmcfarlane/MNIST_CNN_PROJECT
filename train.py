
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import DigitCNN

# Make our experiment reproducible.
torch.manual_seed(42)

# Select GPU if available, otherwise CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Training on: {device}")

# Load the MNIST training dataset.
transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Reserve 10,000 images for validation.
train_data, val_data = random_split(
    dataset,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True
)
val_loader = DataLoader(
    val_data, batch_size=64, shuffle=False
)

# Create the neural network.
model = DigitCNN().to(device)

# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float("inf")
patience = 3
epochs_without_improvement = 0

# Train for up to 10 epochs.
for epoch in range(10):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

    # Evaluate on validation images.
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

    train_loss /= len(train_data)
    val_loss /= len(val_data)
    train_acc = 100 * train_correct / len(train_data)
    val_acc = 100 * val_correct / len(val_data)

    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss={train_loss:.4f}, "
        f"Train Acc={train_acc:.2f}%, "
        f"Val Loss={val_loss:.4f}, "
        f"Val Acc={val_acc:.2f}%",
        flush=True
    )

    # Save the best model and stop if progress stalls.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

print("Training complete. Best model saved.")
