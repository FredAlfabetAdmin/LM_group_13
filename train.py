import torch, glob, os, re, cv2, fire
from torch import nn, optim
import pandas as pd
import numpy as np

def get_img(read_img: str, noise_thresh = 0):
    img = cv2.imread(f'./dataset/images/{read_img}')

    img = cv2.resize(img, (640, 480))
    # Convert the image to HSV color space (Hue, Saturation, Value)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imwrite(str("./frame_org.png"), img)

    # Define the color ranges for green, yellow, and white in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_white = np.array([0, 0, 128])
    upper_white = np.array([255, 50, 255])

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Create masks for each color
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)

    # Set the corresponding color values for each mask
    img[mask_green > 0] = [0, 255, 0]       # Green cubes
    img[mask_yellow > 0] = [0, 0, 0]      # Yellow floor
    img[mask_white > 0] = [255, 255, 255]   # White walls
    img[mask_red > 0] = [0, 0, 255]        # Red items
    
    img = img + (np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * noise_thresh)

    return img

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
            # # Reduce 4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.cnn_output = 128 * (640//16) * (480//16)
        self.lstm_features = 2048 # + 2 # + 2 for binary classification if red or green

        # Connection layer
        self.connection = nn.Linear(self.cnn_output, self.lstm_features)
        self.ln_activation = nn.ReLU()
        self.bn_activation = nn.Sigmoid()

        # LSTM
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(self.lstm_features, lstm_hidden_size, lstm_num_layers, batch_first=True)

        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x: torch.Tensor(), seq: torch.Tensor()): #x is the new image, seq is the previous sequence it gave
        # x, seq = x
        x = self.cnn(x)
        x = x.view(-1, self.cnn_output)  # Adjusted to the new input size
        x = self.connection(x)
        seq = torch.cat([seq[:,1:,:], x.unsqueeze(1)], dim=1)
        x, _ = self.lstm(seq)
        x = x[:, -1, :]
        x = self.fc(x)
        return x, seq

def main(seq_length = 16, batch_size = 16, learning_rate = 0.005, device = 'cuda:0', num_layers = 5):
    model = CNNwithLSTM(num_classes=num_layers)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    for file_ in glob.glob('./dataset/*.csv'):
        round_ = re.findall(r'\d+', file_)[0]
        targets = pd.read_csv(file_)
        seq = torch.zeros([1,seq_length,model.lstm_features], requires_grad=True, device=device)
        try:
            for step, row in targets.iterrows():
                did_optim = False
                img = get_img(read_img = row['image'])
                x = torch.tensor(np.expand_dims(img.swapaxes(-1, 0).swapaxes(-1, 1), 0), dtype=torch.float32, device=device)
                p, seq_new = model(x, seq)
                p = p[0]
                target = torch.tensor(int(row['target']), dtype=torch.long, device=device)
                loss = loss_fn(p, target)
                loss.backward()
                if step % batch_size == 0 and step != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    did_optim = True
                seq = seq_new.detach()
            print(f'round: {round_}, loss: {loss.item()}, target: {target.item()}, direction: {p}')
            if not did_optim:
                optimizer.step()
                optimizer.zero_grad()
        except:
            pass

if __name__ == "__main__":
    fire.Fire(main)