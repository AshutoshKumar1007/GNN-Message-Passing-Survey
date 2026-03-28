import torch
import matplotlib.pyplot as plt
import glob
import os


files = glob.glob("results/*.pt")

for file in files:
    data = torch.load(file)

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    val_acc = data["val_acc"]

    name = os.path.basename(file)

    # Loss plot
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title(f"Loss - {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Accuracy plot
    plt.figure()
    plt.plot(val_acc, label="Val Accuracy")
    plt.legend()
    plt.title(f"Accuracy - {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()