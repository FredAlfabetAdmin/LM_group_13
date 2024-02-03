import torch, glob, os, re, cv2, fire
from torch import nn, optim
import pandas as pd
import numpy as np
from random import shuffle
import json

'''
How to run this?
Just run:
python train.py [args]
these args can be any of the args in the def main. 
So we want to change the name of our model and seq_length then run in the terminal:
python train.py --model_name NAME --seq_length NUMBER 
'''

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
    def __init__(self, num_classes=4, lstm_hidden_size=256, lstm_num_layers=1, bottle_neck_size=2048):
        super().__init__()

        # CNN architecture
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

        # Calculate CNN output size
        self.cnn_output = 128 * (640//16) * (480//16)

        # LSTM features size
        self.lstm_features = bottle_neck_size

        # Connection layer
        self.connection = nn.Linear(self.cnn_output, self.lstm_features)
        self.ln_activation = nn.ReLU()
        self.bn_activation = nn.Sigmoid()

        # LSTM
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(self.lstm_features, lstm_hidden_size, lstm_num_layers, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x: torch.Tensor(), seq: torch.Tensor()):  # x is the new image, seq is the previous sequence it gave
        # Forward pass through CNN
        x = self.cnn(x)
        x = x.view(-1, self.cnn_output)  # Adjusted to the new input size

        # Connection layer
        x = self.connection(x)

        # Concatenate with the previous sequence
        seq = torch.cat([seq[:, 1:, :], x.unsqueeze(1)], dim=1)

        # LSTM layer
        x, _ = self.lstm(seq)
        x = x[:, -1, :]  # Extract the output of the last time step

        # Fully connected layer for classification
        x = self.fc(x)

        return x, seq

def main(model_name = 'dev', seq_length = 16, batch_size = 16, learning_rate = 0.005, num_classes = 5, 
         epochs = 100, device = 'cuda:0', lstm_hidden_size=256, lstm_num_layers=1, bottle_neck_size = 2048, finetune_from = None, checkpoint_num = None):
    # Setup file structure
    os.makedirs('./trainings', exist_ok=True)
    os.makedirs(f'./trainings/{model_name}/checkpoints', exist_ok=True)
    print('found_hparams', locals())

    # If you continue training we use the hyperparameters from that model
    hparams = locals()
    if finetune_from is not None or checkpoint_num is not None:
        assert finetune_from is not None and checkpoint_num is not None, 'Error, forgot to set the checkpoint'

    if finetune_from is not None:
        with open(f'./trainings/{finetune_from}/hparams.json', 'r') as file_:
            new_params = json.load(file_)
        hparams['num_classes'] = new_params['num_classes']
        hparams['lstm_hidden_size'] = new_params['lstm_hidden_size']
        hparams['lstm_num_layers'] = new_params['lstm_num_layers']
        hparams['bottle_neck_size'] = new_params['bottle_neck_size']

    with open(f'./trainings/{model_name}/hparams.json', 'w') as file_:
        json.dump(hparams, file_)
    
    # Define model, loss fn and the optimizer
    model = CNNwithLSTM(num_classes=num_classes, lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers, bottle_neck_size=bottle_neck_size)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    # Load checkpoint if needed
    if checkpoint_num is not None:
        path_ = f'./trainings/{finetune_from}/model.ckpt' if checkpoint_num == -1 else f'./trainings/{finetune_from}/checkpoints/{checkpoint_num}.ckpt'
        model.load_state_dict(torch.load(path_))

    # For storing statistics
    stats_df = {
        'epoch': [],
        'losses': [],
        'rounds': [],
        'target_steps': [],
        'model_prob_0': [],
        'model_prob_1': [],
        'model_prob_2': [],
        'model_prob_3': [],
        'model_prob_4': [],
    }

    # Run x epochs (number of times through the dataset)
    for epoch in range(epochs):
        # For stats
        epoch_loss, epoch_target_steps, epoch_round = [], [], []
        model_probs_0, model_probs_1, model_probs_2, model_probs_3, model_probs_4 = [], [], [], [], []
        # Load csv files from dataset in random order and play them.
        files = glob.glob('./dataset/*.csv')
        shuffle(files)
        for file_ in files: #Iterate over training rounds
            round_ = re.findall(r'\d+', file_)[0]
            targets = pd.read_csv(file_)
            # Init sequence
            seq = torch.zeros([1,seq_length,model.lstm_features], requires_grad=True, device=device)
            # Play the path and let the model learn to take the right actions
            for step, row in targets.iterrows():
                did_optim = False
                # Load image from dataset
                img = get_img(read_img = row['image'])
                x = torch.tensor(np.expand_dims(img.swapaxes(-1, 0).swapaxes(-1, 1), 0), dtype=torch.float32, device=device)
                # Make predictions
                p, seq_new = model(x, seq)
                p = p[0]
                # Calculate loss
                target = torch.tensor(int(row['target']), dtype=torch.long, device=device)
                loss = loss_fn(p, target)
                loss.backward()

                # Stats
                epoch_loss.append(loss.item())
                epoch_target_steps.append(target.item())
                model_probs_0.append(p[0])
                model_probs_1.append(p[1])
                model_probs_2.append(p[2])
                model_probs_3.append(p[3])
                model_probs_4.append(p[4])
                epoch_round.append(round_)

                # Update model very batch_size of steps
                if step % batch_size == 0 and step != 0:
                    optimizer.step()
                    optimizer.zero_grad() #Very important! If we dont do this the gradients are not cleaned and we get gradient leaks
                    did_optim = True
                    print(f'Epoch: {epoch}, step: {step//batch_size}, mean_loss: {np.mean(epoch_loss)}, running loss: {epoch_loss[-1]}')
                # Update the sequence
                seq = seq_new.detach()

            # If we do still need to do an optimizer step, but it is smaller than the batchsize do it anyway
            if not did_optim:
                optimizer.step()
                optimizer.zero_grad() #Very important! If we dont do this the gradients are not cleaned and we get gradient leaks
        
        # More stats bullshit
        stats_df['epoch'].extend([epoch for _ in range(len(epoch_loss))]) #Prepend epoch number
        stats_df['losses'].extend(epoch_loss)
        stats_df['rounds'].extend(epoch_round)
        stats_df['target_steps'].extend(epoch_target_steps)
        stats_df['model_prob_0'].extend(model_probs_0)
        stats_df['model_prob_1'].extend(model_probs_1)
        stats_df['model_prob_2'].extend(model_probs_2)
        stats_df['model_prob_3'].extend(model_probs_3)
        stats_df['model_prob_4'].extend(model_probs_4)

        # Then write the files
        pd.DataFrame.from_dict(stats_df).to_csv(f'./trainings/{model_name}/checkpoints/{epoch}.ckpt', index=False) #Safe file writing, otherwise we could lose data
        torch.save(model.state_dict(), f'./{epoch}.ckpt')

    # Write the final trained model
    torch.save(model.state_dict(), f'./trainings/{model_name}/model.ckpt')
    
if __name__ == "__main__":
    fire.Fire(main)