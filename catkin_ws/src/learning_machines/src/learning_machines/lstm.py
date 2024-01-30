import torch
import math
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output
    
class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Change in_channels to 1
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjusted fully connected layers to accommodate the new input size
        self.fc1 = nn.Linear(64 * (640//8) * (480//8), 128)
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

class CNNwithLSTM(nn.Module):
    def __init__(self, num_classes=4, lstm_hidden_size=256, lstm_num_layers=1):
        super().__init__()
        self.cnn = nn.Sequential(
            # Reduce 1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Reduce 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Reduce 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Reduce 4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.cnn_output = 128 * (640//16) * (480//16)
        self.lstm_features = 512

        # Connection layer
        self.connection = nn.Linear(self.cnn_output, self.lstm_features)
        self.activation = nn.ReLU()

        # LSTM
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(self.lstm_features, lstm_hidden_size, lstm_num_layers, batch_first=True)

        self.fc = nn.Linear(lstm_hidden_size, num_classes)

        # Maybe we can use this fc layer to connect the CNN to the LSTM and reduce the dimensionality of the image.
        # self.fc1 = nn.Linear(64 * (640//8) * (480//8), 128)
        # self.relu4 = nn.ReLU()
        # self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor(), seq: torch.Tensor()): #x is the new image, seq is the previous sequence it gave
        # x, seq = x
        x = self.cnn(x)
        x = x.view(-1, self.cnn_output)  # Adjusted to the new input size
        x = self.connection(x)
        x = self.activation(x)
        seq = torch.cat([seq[:,1:,:], x.unsqueeze(1)], dim=1)
        x, _ = self.lstm(seq)
        x = x[:, -1, :]
        x = self.fc(x)
        return x, seq

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