
import matplotlib.pyplot as plt

# Results recorded from our actual training run.
epochs = list(range(1, 11))

train_loss = [
    0.2608, 0.0759, 0.0532, 0.0455, 0.0377,
    0.0312, 0.0259, 0.0217, 0.0199, 0.0177
]

val_loss = [
    0.0680, 0.0519, 0.0548, 0.0450, 0.0463,
    0.0429, 0.0445, 0.0401, 0.0390, 0.0441
]

train_acc = [
    91.81, 97.72, 98.32, 98.58, 98.81,
    99.02, 99.14, 99.28, 99.36, 99.45
]

val_acc = [
    98.07, 98.56, 98.32, 98.77, 98.71,
    98.81, 98.76, 98.92, 99.04, 98.96
]

# Plot accuracy.
plt.figure(figsize=(9, 5))
plt.plot(epochs, train_acc, marker="o", label="Training Accuracy")
plt.plot(epochs, val_acc, marker="s", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_accuracy.png", dpi=200)
plt.show()

# Plot loss.
plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.plot(epochs, val_loss, marker="s", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=200)
plt.show()

print("Training graphs saved successfully.")
