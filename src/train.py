print("Training script started...")
import torch #   pytorch linrary It gives you tensors (like NumPy arrays but faster & can use GPU).
#It also has math operations, autograd (automatic differentiation), and training utilities.
import torch.nn as nn #  neural network module in PyTorch. It provides classes and functions to 
# build and train neural networks
import torch.nn.functional as F #  functional module in PyTorch. It provides functions for
# common neural network operations, such as activation functions, loss functions, and pooling.
#nn gives you ready-made layers (e.g., nn.ReLU())
#F gives you the math version (e.g., F.relu(x))
import torch.optim as optim # optimization module in PyTorch. It provides various optmization
# algorithms to update model parameters during training.
from torch.utils.data import DataLoader # data module in PyTorch. It provides utilities for loading
# and processing datasets, including data loaders and dataset classes.
#datasets → gives you standard datasets (MNIST, CIFAR10, etc.).
#transforms → lets you preprocess images (resize, normalize, convert to tensor).
#DataLoader → takes a dataset and makes it easy to grab batches (instead of one image at a time).
#from torchvision import datasets,transforms # torchvision is a package in PyTorch that provides
# datasets,models,and image tranformations for computer vision tasks,
import torchvision.transforms as transforms # transforms module in torchvision. It provides functions
from torchvision import datasets #gives readymade datasets
# to apply common image transformations, such as resizing, cropping, and normalization.
#define data and labels
#data = [[1,2] , [3,4], [5,6]]
#labels = [0 , 1 , 0]
from model import SimpleCNN
# hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 5

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307),(0.3081))]) # takes raw imgae and converts it into a tensor to floats in range [0,1]
train_dataset = datasets.MNIST(root = "data", train= True, download = True, transform = transform)
#dataset  = CustomDataset(data,labels)               # create an instance of the CustomDataset class with data and labels
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True) # create a DataLoader instance with the dataset, batch size, and shuffle option
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
print("train size:", len(train_dataset))
print("test size:", len(test_dataset))
for epoch in range(epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
      #forward
        logits = model(images)
        loss = criterion(logits,labels)
      #backward
        optimizer.zero_grad()
        loss.backward()
      # update
        optimizer.step()
      # track loss
        running_loss += loss.item()
      avg_loss = running_loss / len(train_loader)
      print(f"Epoch {epoch+1}/{epochs} - train loss: {avg_loss: .4f}")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total if total > 0 else 0.0
print(f"Test Accuracy: {accuracy:.2f}%")
torch.save(model.state_dict(), "models/mnist_cnn.pth")
print("Model saved to models/mnist_cnn.pth")




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,data, labels): # CustomDataset is a class that inherits from torch.utils.data.Dataset.
        self.data = torch.tensor(data, dtype = torch.float32) # initialize the dataset with data and labels
        self.labels = torch.tensor(labels, dtype = torch.long) # intialize the dataset with data and labels
       
       
    def  __len__(self):
            return len(self.data) # returns the length of the dataset
       
    def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]# returns the data and label at the given index



#for batch_data, batch_labels in dataLoader:
 #     print(batch_data,batch_labels)   # iterate over the DataLoader and print the batch data and labels
#print(len(dataset))
#print(dataset[0])
images,labels = next(iter(train_loader))
#iter(train_loader) → makes an iterator object from your DataLoader.
#next(iter(train_loader)) → asks that iterator for the next batch (the first one if it’s new).
#print("images:", images.shape, images.dtype)  # expect: torch.Size([32, 1, 28, 28]) float32
#print("labels:",labels.shape,labels.dtype) # expect: torch.Size([32])             int64
#logits = model(images)               # forward pass
#print("logits:", logits.shape)       # expect: [batch_size, 10]logits = model(images)               # forward pass
