
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from model import DigitCNN

# Use the same device as training.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load the separate MNIST test dataset.
test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_data, batch_size=64, shuffle=False
)

# Restore the best saved CNN.
model = DigitCNN().to(device)
model.load_state_dict(
    torch.load("best_model.pth", map_location=device, weights_only=True)
)
model.eval()

actual = []
predicted = []

# Generate predictions without updating model weights.
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu()

        actual.extend(labels.tolist())
        predicted.extend(predictions.tolist())

# Print test performance.
print(f"Test Accuracy: {accuracy_score(actual, predicted)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(actual, predicted, digits=4))

# Visualize and save the confusion matrix.
matrix = confusion_matrix(actual, predicted)
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=list(range(10))
)

fig, ax = plt.subplots(figsize=(9, 8))
display.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("MNIST Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()
