import torch
import math
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output
    
class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Change in_channels to 1
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjusted fully connected layers to accommodate the new input size
        self.fc1 = nn.Linear(64 * (430//8) * (680//8), 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = x.view(-1, 64 * (640//8) * (480//8))  # Adjusted to the new input size
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

class NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = x[-1]
        x = self.linear(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x

def eucl_loss_fn(point1, point2):
    delta1 = point1[0] - point2[0]
    delta2 = point1[1] - point2[1]
    delta1 = torch.pow(delta1, 2)
    delta2 = torch.pow(delta2, 2)
    total = delta1 + delta2
    distance = torch.sqrt(total)
    return distance

# network = LSTM()
# network.train()
# loss_fn = eucl_loss_fn

'''    
    prediction_batch = network(X_batch)  # forward pass
    print(prediction_batch.size(), Y_batch.size())
    batch_loss = loss_fn(prediction_batch.squeeze(), Y_batch)  # loss calculation
    batch_loss.backward()  # gradient computation
    optimizer.step()  # back-propagation
    optimizer.zero_grad()  # gradient reset
    return batch_loss.item()
'''