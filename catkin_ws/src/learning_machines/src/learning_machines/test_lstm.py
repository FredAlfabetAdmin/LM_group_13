from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
)
from robobo_interface.datatypes import (
    Acceleration,
    Position,
    Orientation,
)

from .lstm import eucl_loss_fn, LSTM
from torch import optim, nn
import torch, time, random
import numpy as np

def move_robobo(movement, rob):
    movement = movement.detach().numpy() #Use clone since detach doesnt properly work
    movement = np.argmax(movement)
    if movement == 0: #Move forward
        movement = [50, 50, 250]
    elif movement == 1: #Move backward
        movement = [-50, -50, 250]
    elif movement == 2: #Move left
        movement = [-50, 50, 250]
    elif movement == 3: #Move right
        movement = [50, -50, 250]
    rob.move_blocking(int(movement[0]), int(movement[1]), int(movement[2]))

class FoodDetect():
    def __init__(self):
        '''class to see if food is found and decay this over time'''
        max_len = 10
        self.highest_num = 0
        self.food = torch.zeros((max_len,), dtype=torch.float32, requires_grad=True)
        self.mask = torch.tensor([1-(x*(1/(max_len-1))) for x in range(max_len)], dtype=torch.float32, requires_grad=True)
        tot_ = torch.sum(self.mask.detach())

    def add_food(self, food: int) -> int:
        # Add nr_food to list and apply the mask
        food -= self.highest_num #If found one it gets increased by 1 or 2 otherwise it will be 0
        self.food = self.food[1:] #Tick the buffer one over
        self.food = torch.cat([self.food, food])
        if self.highest_num > food[0]:
            self.highest_num = food[0]
        print('food_calc', self.food * self.mask)
        return (-torch.sum(self.food * self.mask) + 1 ) * 10 #Apply the mask and sum.

class RobFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rob):
        ctx.save_for_backward(input)
        move_robobo(input, rob)
        food_and_time = torch.tensor([rob.nr_food_collected(), rob.get_sim_time()], dtype=torch.float32, requires_grad=True)
        return food_and_time

    @staticmethod
    def backward(ctx, grad_output):
        # ctx.saved_tensors contains the input from the forward pass
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.full((4,), grad_output[1]) #Use 1 for the rob_pos grad, *100 since the output of the model is like that
        return grad_input, None

def run_lstm_classification(rob: IRobobo):
    with torch.autograd.detect_anomaly():
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()
        print('connected')
        # Setup things
        max_time = 3*60*1000
        time_alpha = 100

        X_size = 14
        seq_len = 8
        seq = torch.zeros((1, seq_len, X_size), dtype=torch.float32)
        # Define the model and set it into train mode, together with the optimizer
        network = LSTM(X_size, 32, 4, 2)
        # Eval model in hw
        if not isinstance(rob, SimulationRobobo):
            network.load_state_dict(torch.load('./model.ckpt'))
            network.eval()
            with torch.no_grad():
                for _ in range(200): #Take max 75 steps per round
                    robfn = RobFN.apply #Define the custom grad layer
                    # Get the input data
                    orientation = rob.read_orientation()
                    accelleration = rob.read_accel()
                    x = torch.tensor([x*100000 for x in rob.read_irs()] + [orientation.yaw, orientation.pitch, orientation.roll] + [accelleration.x, accelleration.y, accelleration.z], dtype=torch.float32)
                    seq = torch.cat([seq[:, 1:, :], x.unsqueeze(0).unsqueeze(0)], dim=1)
                    p = network(seq) #Do the forward pass
                    p = nn.functional.softmax(p, dim=0)
                    move_robobo(p, rob)
            return
        
        optimizer = optim.Adam(params=network.parameters(), lr=0.05)
        network.train()
        print('Started training')
        for round_ in range(20): #Reset the robobo and target 10 times with or without random pos
            rob.play_simulation()
            for step in range(240): #Take max 75 steps per round
                robfn = RobFN.apply #Define the custom grad layer
                orientation = rob.read_orientation()
                accelleration = rob.read_accel()
                x = torch.tensor(rob.read_irs() + [orientation.yaw, orientation.pitch, orientation.roll] + [accelleration.x, accelleration.y, accelleration.z], dtype=torch.float32)
                seq = torch.cat([seq[:, 1:, :], x.unsqueeze(0).unsqueeze(0)], dim=1)
                p = network(seq) #Do the forward pass
                p = p[0, -1, :]
                p = nn.functional.softmax(p, dim=0)
                food_and_time = robfn(p, *(rob,)) #Calculate the custom robotics gradients

                if food_and_time[0] > 0:
                    n_food_reward = -torch.ceil(food_and_time[0]*(torch.log10(food_and_time[0]))) + 7
                else:
                    n_food_reward = food_and_time[0] + 7
                sim_time = ((torch.pow(food_and_time[1], 2)*(1/max_time))) / max_time * time_alpha
                loss =  n_food_reward + sim_time 

                print(f'round: {round_}, loss: {loss.item()}')
                loss.backward() #Do the backward pass
                optimizer.step() #Do a step in the learning space
                optimizer.zero_grad() #Clear the gradients in the optimizer
                if food_and_time[0] >= 7 or sim_time > 3*60*1000:
                    print('object_completed')
                    break
            rob.stop_simulation()
            time.sleep(0.25)

        torch.save(network.state_dict(), './model.ckpt')

        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()