import torch
import math
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int, b_size: int):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.h_to_y = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Tanh()
        self.hidden = torch.zeros((output_dim, b_size, hidden_dim))
        self.cstate = torch.zeros((output_dim, b_size, hidden_dim))
    
    def forward(self, text):
        # text dim: [sentence length, batch size]
        embedded = self.embedding(text)
        # embedded dim: [sentence length, batch size, embedding dim]
        embedded = torch.swapaxes(embedded, 0, 1)
        output, (hidden, cstate) = self.rnn(embedded, (self.hidden, self.cstate))
        self.hidden = hidden
        self.cstate = cstate
        output = torch.swapaxes(output, 0, 1)
        output = self.h_to_y(output)
        output = self.output_activation(output)
        return output

class NN(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int, b_size: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
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