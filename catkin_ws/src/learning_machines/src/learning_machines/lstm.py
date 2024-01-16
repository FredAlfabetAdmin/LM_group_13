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
    

def eucl_loss_fn(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
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