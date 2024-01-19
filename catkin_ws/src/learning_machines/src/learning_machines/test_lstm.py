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
    def forward(ctx, input, rob):
        ctx.save_for_backward(input)
        move_robobo(input, rob)
        rob_pos = rob.position()

        rob_pos = torch.tensor([rob_pos.x, rob_pos.y], dtype=torch.float32, requires_grad=True)
        
        return rob_pos

    @staticmethod
    def backward(ctx, grad_output):
        # ctx.saved_tensors contains the input from the forward pass
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.full((1,), grad_output[1]) #Use 1 for the rob_pos grad, *100 since the output of the model is like that
        return grad_input, None

def stop_check(rob, target):
    x_diff = rob[0] - target[0]
    y_diff = rob[1] - target[1]
    if abs(x_diff) < 0.16 and abs(y_diff) < 0.16:
        return True

def run_lstm_regression(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print('connected')
    # Define the model and set it into train mode, together with the optimizer
    network = LSTM(14, 128, 16, 4, 1)
    loss_fn = eucl_loss_fn
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
                p = network(x) #Do the forward pass
                p = torch.argmax(nn.functional.softmax(p, dim=0))
                move_robobo(p, rob)
        return
    network.train()
    optimizer = optim.Adam(params=network.parameters(), lr=0.01)

    print('Started training')
    for round_ in range(20): #Reset the robobo and target 10 times with or without random pos
        # if rnd_pos:
        if round_ > 3: #Only create random positions after 3 rounds
            rob.set_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.03), Orientation(-90, -90, -90))
            rob.set_target_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.2))
        rob.play_simulation()
        start = time.time()
        for _ in range(150): #Take max 75 steps per round
            optimizer.zero_grad() #Clear the gradients in the optimizer
            robfn = RobFN.apply #Define the custom grad layer
            # Get the input data
            orientation = rob.read_orientation()
            accelleration = rob.read_accel()
            # Make the input tensor (or the input data)
            x = torch.tensor(rob.read_irs() + [orientation.yaw, orientation.pitch, orientation.roll] + [accelleration.x, accelleration.y, accelleration.z], dtype=torch.float32)
            
            p = network(x) #Do the forward pass
            # # Take the wheel output and map it to neg to pos and the time to sigmoid.
            p = torch.concat([torch.unsqueeze(nn.functional.tanh(p[0]), 0), torch.unsqueeze(nn.functional.tanh(p[1]), 0), torch.unsqueeze(nn.functional.sigmoid(p[2]), 0)])
            p[0] = torch.trunc(p[0]*100) #Multiply the output of the model (as regression now) and truncate
            p[1] = torch.trunc(p[1]*100) #Multiply the output of the model (as regression now) and truncate
            p[2] = torch.trunc(p[2]*500) #Multiply the output of the model (as regression now) and truncate
            
            rob_pos = robfn(p, *(rob,)) #Calculate the custom robotics gradients
            # Get the positions and make them tensors
            target_pos = rob.get_target_position()
            target_pos = torch.tensor([target_pos.x, target_pos.y], dtype=torch.float32, requires_grad=True)
            loss = torch.pow(loss_fn(target_pos, rob_pos), 2) #Calculate the euclidean distance
            # loss = loss + ((time.time() - start)*0.15) #Could be used to give a penalty for time
            print(f'round: {round_}\n, loss: {loss.item()}\n{p}\n')
            loss.backward() #Do the backward pass
            optimizer.step() #Do a step in the learning space
        rob.stop_simulation()
        time.sleep(0.25)

    torch.save(network.state_dict(), './model.ckpt')

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

def run_lstm_classification(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print('connected')
    X_size = 14
    seq_len = 8
    # Define the model and set it into train mode, together with the optimizer
    network = LSTM(X_size, 32, 4, 2)
    loss_fn = eucl_loss_fn
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
                p = network(x) #Do the forward pass
                p = torch.argmax(nn.functional.softmax(p, dim=0))
                move_robobo(p, rob)
        return
    network.train()
    optimizer = optim.Adam(params=network.parameters(), lr=0.05)

    seq = torch.zeros((1, seq_len, X_size), dtype=torch.float32)
    batch_size = 16

    print('Started training')
    for round_ in range(20): #Reset the robobo and target 10 times with or without random pos
        # if rnd_pos:
        if round_ > 3: #Only create random positions after 3 rounds
            rob.set_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.03), Orientation(-90, -90, -90))
            rob.set_target_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.2))
        rob.play_simulation()
        start = time.time()
        optimizer.zero_grad() #Clear the gradients in the optimizer
        for step in range(240): #Take max 75 steps per round
            robfn = RobFN.apply #Define the custom grad layer
            # Get the input data
            orientation = rob.read_orientation()
            accelleration = rob.read_accel()
            # Make the input tensor (or the input data)
            x = torch.tensor(rob.read_irs() + [orientation.yaw, orientation.pitch, orientation.roll] + [accelleration.x, accelleration.y, accelleration.z], dtype=torch.float32)
            seq = torch.cat([x.unsqueeze(0).unsqueeze(0), seq[:, :-1, :]], dim=1)
            p = network(seq) #Do the forward pass
            p = p[0, -1, :]
            p = torch.argmax(nn.functional.softmax(p, dim=0))
            # # Take the wheel output and map it to neg to pos and the time to sigmoid.
            # p = torch.concat([torch.unsqueeze(nn.functional.tanh(p[0]), 0), torch.unsqueeze(nn.functional.tanh(p[1]), 0), torch.unsqueeze(nn.functional.sigmoid(p[2]), 0)])
            # p[0] = torch.trunc(p[0]*100) #Multiply the output of the model (as regression now) and truncate
            # p[1] = torch.trunc(p[1]*100) #Multiply the output of the model (as regression now) and truncate
            # p[2] = torch.trunc(p[2]*500) #Multiply the output of the model (as regression now) and truncate
            
            rob_pos = robfn(p, *(rob,)) #Calculate the custom robotics gradients
            # Get the positions and make them tensors
            target_pos = rob.get_target_position()                
            target_pos = torch.tensor([target_pos.x, target_pos.y], dtype=torch.float32, requires_grad=True)
            loss = loss_fn(target_pos, rob_pos) #Calculate the euclidean distance
            loss = loss + ((time.time() - start)*0.001) #Could be used to give a penalty for time
            print(f'round: {round_}\n, loss: {loss.item()}\n{p}\n')
            loss.backward() #Do the backward pass
            if stop_check(rob_pos.detach().numpy(), target_pos.detach().numpy()):
                print('object_completed')
                optimizer.step() #Do a step in the learning space
                optimizer.zero_grad() #Clear the gradients in the optimizer
                break
            if step % batch_size == 0 and step != 0:
                optimizer.step() #Do a step in the learning space
                optimizer.zero_grad() #Clear the gradients in the optimizer
        rob.stop_simulation()
        time.sleep(0.25)

    torch.save(network.state_dict(), './model.ckpt')

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()