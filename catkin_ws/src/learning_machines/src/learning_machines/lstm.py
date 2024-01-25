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

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])  # Take the output from the last time step
#         return out

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