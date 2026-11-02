
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Convert images into PyTorch tensors.
transform = transforms.ToTensor()

# Download the MNIST training dataset.
train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Check the number of images.
print(f"Number of training images: {len(train_data)}")

# Display the first nine handwritten digits.
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
    image, label = train_data[i]

    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(f"Digit: {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()
