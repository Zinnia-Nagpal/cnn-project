print("Testing script started...")

import torch
from torchvision import datasets, transforms
from model import SimpleCNN
from utils import load_model, evaluate, show_image

# reload model
model = load_model(SimpleCNN, "models/mnist_cnn.pth")

# test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# evaluate
evaluate(model, test_loader)

# single prediction
sample_img, sample_label = test_dataset[0]
with torch.no_grad():
    pred = model(sample_img.unsqueeze(0)).argmax(dim=1).item()


# single prediction
sample_img, sample_label = test_dataset[0]
with torch.no_grad():
    pred = model(sample_img.unsqueeze(0)).argmax(dim=1).item()

# save preview image
show_image(sample_img, label=sample_label, pred=pred, save_path="preview.png")

