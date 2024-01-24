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
import joblib

scaler = joblib.load('software_powertrans_scaler.gz')

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

class RobFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, rob):
        ctx.save_for_backward(inp)
        move_robobo(inp, rob)
        irs = torch.tensor(rob.read_irs(), dtype=torch.float32, requires_grad=True)
        return irs

    @staticmethod
    def backward(ctx, grad_output):
        # ctx.saved_tensors contains the input from the forward pass
        inp, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.full_like(inp, torch.mean(grad_output))
        return grad_input, None

def obstcl_avoid_loss(irs):
    return torch.sum(irs)

def run_lstm_classification(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print('connected')
    X_size = 12
    seq_len = 8
    # Define the model and set it into train mode, together with the optimizer
    network = LSTM(X_size, 32, 4, 2)
    loss_fn = obstcl_avoid_loss
    seq = torch.zeros((1, seq_len, X_size), dtype=torch.float32)
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
                x = torch.tensor(rob.read_irs() + [orientation.yaw] + [accelleration.x, accelleration.y, accelleration.z], dtype=torch.float32)
                seq = torch.cat([x.unsqueeze(0).unsqueeze(0), seq[:, :-1, :]], dim=1)
                p = network(seq) #Do the forward pass
                p = torch.argmax(nn.functional.softmax(p, dim=0))
                move_robobo(p, rob)
        return
    network.train()
    optimizer = optim.Adam(params=network.parameters(), lr=0.05)

    print('Started training')
    for round_ in range(20): #Reset the robobo and target 10 times with or without random pos
        # if rnd_pos:
        if round_ > 3: #Only create random positions after 3 rounds
            rob.set_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.03), Orientation(-90, -90, -90))
            rob.set_target_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.2))
        rob.play_simulation()
        for _ in range(240): #Take max 75 steps per round
            optimizer.zero_grad() #Clear the gradients in the optimizer
            robfn = RobFN.apply #Define the custom grad layer
            # Get the input data
            orientation = rob.read_orientation()
            accelleration = rob.read_accel()
            
            irs_inp = scaler.transform([rob.read_irs()])[0].tolist()
            # Make the input tensor (or the input data)
            x = torch.tensor(irs_inp + [orientation.yaw] + [accelleration.x, accelleration.y, accelleration.z], dtype=torch.float32)
            seq = torch.cat([seq[:, 1:, :], x.unsqueeze(0).unsqueeze(0)], dim=1)
            p = network(seq) #Do the forward pass
            p = p[0, -1, :]
            p = nn.functional.softmax(p, dim=0)
            irs = robfn(p, *(rob,))
            loss = loss_fn(irs) #Calculate the euclidean distance
            print(f'round: {round_}, loss: {loss.item()}\n')
            loss.backward() #Do the backward pass
            optimizer.step() #Do a step in the learning space
        rob.stop_simulation()
        time.sleep(0.25)

    torch.save(network.state_dict(), './model.ckpt')

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()