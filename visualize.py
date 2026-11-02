
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import DigitCNN

# Load our trained CNN.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DigitCNN().to(device)
model.load_state_dict(
    torch.load("best_model.pth", map_location=device, weights_only=True)
)
model.eval()

# Load one MNIST test image.
test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

image, label = test_data[0]
input_image = image.unsqueeze(0).to(device)

# Visualize the first convolutional layer's learned filters.
filters = model.conv1.weight.detach().cpu()

fig, axes = plt.subplots(4, 8, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i, 0], cmap="gray")
    ax.axis("off")

plt.suptitle("Learned Filters - First Convolutional Layer")
plt.tight_layout()
plt.savefig("cnn_filters.png", dpi=200)
plt.show()

# Extract feature maps from the first convolutional layer.
with torch.no_grad():
    feature_maps = torch.relu(model.conv1(input_image))

feature_maps = feature_maps.squeeze(0).cpu()

fig, axes = plt.subplots(4, 8, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[i], cmap="gray")
    ax.axis("off")

plt.suptitle(f"CNN Feature Maps - True Digit: {label}")
plt.tight_layout()
plt.savefig("cnn_feature_maps.png", dpi=200)
plt.show()

print("Visualizations saved successfully.")
