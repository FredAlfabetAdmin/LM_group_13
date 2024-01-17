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

class RobFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rob):
        ctx.save_for_backward(input)
        movement = input.detach().numpy() #Use clone since detach doesnt properly work
        rob.move_blocking(int(movement[0]), int(movement[1]), int(movement[2]))
        rob_pos = rob.position()

        rob_pos = torch.tensor([rob_pos.x, rob_pos.y], dtype=torch.float32, requires_grad=True)
        
        return rob_pos

    @staticmethod
    def backward(ctx, grad_output):
        # ctx.saved_tensors contains the input from the forward pass
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.full((3,), grad_output[1]) #Use 1 for the rob_pos grad, *100 since the output of the model is like that
            # grad_input = grad_output[1].clone()
            print(grad_input, input)
        return grad_input, None

def run_lstm_sim(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print('connected')
    # Define the model and set it into train mode, together with the optimizer
    network = LSTM(14, 128, 16, 3, 1)
    network.train()
    loss_fn = eucl_loss_fn
    optimizer = optim.Adam(params=network.parameters(), lr=0.1)
    rnd_pos = True

    print('Started training')
    for round_ in range(10): #Reset the robobo and target 10 times with or without random pos
        # if rnd_pos:
        if round_ > 5: #Only create random positions after 5 rounds
            rob.set_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.03), Orientation(-90, -90, -90))
            rob.set_target_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.2))
        rob.play_simulation()
        start = time.time()
        for _ in range(50): #Take max x steps per round
            optimizer.zero_grad() #Clear the gradients in the optimizer
            robfn = RobFN.apply #Define the custom grad layer
            # Get the input data
            orientation = rob.read_orientation()
            accelleration = rob.read_accel()
            # Make the input tensor (or the input data)
            x = torch.tensor(rob.read_irs() + [orientation.yaw, orientation.pitch, orientation.roll] + [accelleration.x, accelleration.y, accelleration.z], dtype=torch.float32)
            
            p = network(x) #Do the forward pass

            # Take the wheel output and map it to neg to pos and the time to sigmoid.
            p = torch.concat([torch.unsqueeze(nn.functional.tanh(p[0]), 0), torch.unsqueeze(nn.functional.tanh(p[1]), 0), torch.unsqueeze(nn.functional.sigmoid(p[2]), 0)])
            p[0] = torch.trunc(p[0]*100) #Multiply the output of the model (as regression now) and truncate
            p[1] = torch.trunc(p[1]*100) #Multiply the output of the model (as regression now) and truncate
            p[2] = torch.trunc(p[2]*500) #Multiply the output of the model (as regression now) and truncate
            
            rob_pos = robfn(p, *(rob,)) #Calculate the custom robotics gradients
            # Get the positions and make them tensors
            target_pos = rob.get_target_position()
            target_pos = torch.tensor([target_pos.x, target_pos.y], dtype=torch.float32, requires_grad=True)
            loss = loss_fn(target_pos, rob_pos) #Calculate the euclidean distance
            # loss = loss + ((time.time() - start)*0.15) #Could be used to give a penalty for time
            print(f'round: {round_}, {loss.item()}\n{p}\n')
            loss.backward() #Do the backward pass
            optimizer.step() #Do a step in the learning space
        rob.stop_simulation()
        time.sleep(0.25)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()