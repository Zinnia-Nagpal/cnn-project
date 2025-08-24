import torch
import matplotlib.pyplot as plt

def save_model(model, path="models/mnist_cnn.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path="models/mnist_cnn.pth"):
    model = model_class()
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f" Model loaded from {path}")
    return model

def evaluate(model, data_loader, device="cpu"):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"📊 Accuracy: {accuracy:.2f}%")
    return accuracy

def show_image(img, label=None, pred=None, save_path=None):
    plt.imshow(img.squeeze(), cmap="gray")
    title = ""
    if label is not None:
        title += f"Label: {label} "
    if pred is not None:
        title += f"Pred: {pred}"
    if title:
        plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
        print(f" Image saved to {save_path}")
        plt.close()
    else:
        plt.show()

