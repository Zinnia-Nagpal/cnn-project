import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

    #input: [B , 1 , 28,28]

        self.conv1 = nn.Conv2d(1,32, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding= 1 )
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # [B,32,28,28]
        x = self.pool(x)              # [B,32,14,14]
        x = F.relu(self.conv2(x))     # [B,64,14,14]
        x = self.pool(x)              # [B,64,7,7]
        x = x.view(x.size(0), -1)     # flatten -> [B, 64*7*7]
        x = F.relu(self.fc1(x))       # [B,128]
        x = self.fc2(x)               # [B,10] (raw scores = logits)
        return x
