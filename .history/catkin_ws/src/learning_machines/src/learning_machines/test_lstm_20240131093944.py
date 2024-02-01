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

from .lstm import *
from torch import optim, nn
import torch, time, random
import numpy as np
import joblib
import cv2
import math
import os

def eucl_fn(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def get_img(rob: IRobobo, noise_thresh = 0):
    img = rob.get_image_front()

    img = cv2.resize(img, (640, 480))
    # Convert the image to HSV color space (Hue, Saturation, Value)
    # cv2.imwrite(str("./frame_org.png"), img)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the color ranges for green, yellow, and white in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_white = np.array([0, 0, 128])
    upper_white = np.array([255, 50, 255])

    # Create masks for each color
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

    # Set the corresponding color values for each mask
    img[mask_green > 0] = [0, 255, 0]       # Green cubes
    img[mask_yellow > 0] = [0, 0, 0]      # Yellow floor
    img[mask_white > 0] = [255, 255, 255]   # White walls
    
    img = img + (np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * noise_thresh)

    cv2.imwrite(str("./frame.png"), img)

    # Create a mask for the green color
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image and calculate centroids
    points = []
    for contour in contours:
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Draw the centroid on the image
            # cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)  # Red circle
            points.append([cx, cy])

    return img, points   

def move_robobo(movement, rob):
    movement = nn.functional.softmax(movement.detach(), dim=0)
    movement = movement.detach().numpy() #Use clone since detach doesnt properly work
    # move_pr = np.argmax(movement)
    move_pr = np.random.choice([0,1,2,3], p=movement)
    if move_pr == 0: #Move forward
        rob.move_blocking(50, 50, 250)
    elif move_pr == 1: #Move backward
        rob.move_blocking(-25, -60, 250)
    elif move_pr == 2: #Move left
        rob.move_blocking(25, 50, 250)
    elif move_pr == 3: #Move right
        rob.move_blocking(50, 25, 250)
    return move_pr

def evaluation(rob, model: nn.Module):
    model.load_state_dict(torch.load('./model_33.ckpt'))
    model.eval()
    
    robot_locations = []
    actions = []
    losses = []
    foods_collected = []
    foods = []

    directory = "./eval_data/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    with open(f'{directory}eval_settings.txt', "w+") as file_:
        file_.writelines([get_xyz(rob.position())])
    seq_length = 8
    with torch.no_grad():
        rob.set_phone_tilt_blocking(105, 100) #Angle phone forward
        seq = torch.zeros([1,seq_length,model.lstm_features])
        while True:
            # Get the input data
            img_, points = get_img(rob)
            x = torch.tensor(np.expand_dims(img_.swapaxes(-1, 0).swapaxes(-1, 1), 0), dtype=torch.float32)
            p, seq = model(x, seq) #Do the forward pass
            p = p[0]
            #move_robobo(p, rob)

            action_taken = move_robobo(p, rob)
            loss = calc_loss_eval(points, p)
            
            
            foods_collected.append(str(rob.nr_food_collected()) + " ")
            robot_locations.append(get_xyz(rob.position()) + " ")
            actions.append(str(action_taken) + " ")
            losses.append(str(loss.item()) + " ")

        #print(losses)
        with open(f'{directory}eval_loss.txt', "w+") as file_:
            file_.writelines(losses)
        
        #print(foods_collected)
        with open(f'{directory}eval_food.txt', "w+") as file_:
            file_.writelines(foods_collected)
        
        #print(actions)
        with open(f'{directory}eval_actions.txt', "w+") as file_:
            file_.writelines(actions)
        
        #print(robot_locations)
        with open(f'{directory}eval_robot_locations.txt', "w+") as file_:
            file_.writelines(robot_locations)

def train(rob, model: nn.Module, optimizer: torch.optim.Optimizer) -> nn.Module:
    print('Started training')
    model.train()
    optimizer.zero_grad()
    loss_fn = nn.CrossEntropyLoss()
    all_loss, all_actions, all_food, all_target = [], [], [], []
    seq_length = 16
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / 500, 1.0))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: max(-math.sqrt(i*(1/200))+2, 1.0))
    for round_ in range(50): #Reset the robobo and target 10 times with or without random pos
        rob.play_simulation()
        start = time.time()
        rob.set_phone_tilt_blocking(105, 100) #Angle phone forward
        loss_am, actions, food, target_am = [], [], [], []
        seq = torch.zeros([1,seq_length,model.lstm_features], requires_grad=True)
        # Set a random position of the robobo every round
        rob.set_position(Position(-2.25-(random.random()*1.75), (random.random()*1.6)-0.048, 0.075), Orientation(-90, -90, -90))
        for step in range(100): # Keep going unless 3 minutes is reached or all food is collected
            did_optim = False
            img_, points = get_img(rob, noise_thresh=0.2)
            x = torch.tensor(np.expand_dims(img_.swapaxes(-1, 0).swapaxes(-1, 1), 0), dtype=torch.float32)
            p, seq_new = model(x, seq) #Do the forward pass
            p = p[0]
            move = move_robobo(p, rob)
            target = -1
            width, height, center = 640, 480, 150
            if len(points) > 0:
                points_right, points_left = 0, 0
                for point in points:
                    # if eucl_fn(point, [640//2, 480//2]) < center: #If near center ignore the rest and continue forward
                    if point[0] >= (width//2-center//2) and point[0] <= (width//2+center//2): #If near center ignore the rest and continue forward
                        target = 0
                        break
                    if point[0] <= (width//2-center//2): #If on the left
                        points_left+=1
                    else:
                        points_right+=1
                if target != 0:
                    if points_left >= points_right: #If more points on the left move left.
                        target = 2
                    else:
                        target = 3
            else: #If not near, just go backward, most likely in the wall
                target = 1

            target = torch.tensor(target, dtype=torch.long)
            loss = loss_fn(p, target)

            loss.backward() #Do the backward pass
            if step % 4 == 0 and step != 0:
                optimizer.step() #Do a step in the learning space
                optimizer.zero_grad() #Clear the gradients in the optimizer
                # scheduler.step() #Increase the learning rate
                did_optim = True
            seq = seq_new.detach()

            print(f'round: {round_}, loss: {loss.item()}, nr_food: {rob.nr_food_collected()}, target: {target.item()}, direction: {move}, learning_rate: {optimizer.param_groups[0]["lr"]}')
            loss_am.append(str(loss.item()))
            actions.append(str(move))
            food.append(str(rob.nr_food_collected()))
            target_am.append(str(target.item()))
            if rob.nr_food_collected() >= 7 or time.time() - start > 3*60*1000:
                print(f'object_completed within time: {time.time() - start}, collected: {rob.nr_food_collected()}')
                break
        if not did_optim:
            optimizer.step() #Do a step in the learning space
            optimizer.zero_grad() #Clear the gradients in the optimizer
            # scheduler.step() #Increase the learning rate
        all_loss.append(' '.join(loss_am))
        all_actions.append(' '.join(actions))
        all_food.append(' '.join(food))
        all_target.append(' '.join(target_am))
        with open(f'./res_loss_{round_}.txt', "w") as file_:
            file_.writelines(all_loss[-1])
        with open(f'./res_food_{round_}.txt', "w") as file_:
            file_.writelines(all_food[-1])
        with open(f'./res_target_{round_}.txt', "w") as file_:
            file_.writelines(all_target[-1])
        with open(f'./res_action_{round_}.txt', "w") as file_:
            file_.writelines(all_actions[-1])
        torch.save(model.state_dict(), f'./model_{round_}.ckpt')
        
        rob.stop_simulation()
        time.sleep(0.25)
    with open(f'./res_loss.txt', "w") as file_:
        file_.writelines(all_loss)
    with open(f'./res_food.txt', "w") as file_:
        file_.writelines(all_food)
    with open(f'./res_target.txt', "w") as file_:
        file_.writelines(all_target)
    with open(f'./res_action.txt', "w") as file_:
        file_.writelines(all_actions)
    return model

def run_lstm_classification(
        rob: IRobobo, 
        num_outputs=4, eval_=True):
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    print('connected')
    if False:
        while True:
            rob.set_phone_tilt_blocking(105, 100) #Angle phone forward
            img, points = get_img(rob)
            target = -1
            width, height, center = 640, 480, 100
            print(points)
            if len(points) > 0:
                points_right, points_left = 0, 0
                for point in points:
                    # if eucl_fn(point, [640//2, 480//2]) < center: #If near center ignore the rest and continue forward
                    if point[0] >= (width//2-center//2) and point[0] <= (width//2+center//2): #If near center ignore the rest and continue forward
                        target = 0
                        rob.move_blocking(50,50,250)
                        break
                    if point[0] <= (width//2-center//2): #If on the left
                        points_left+=1
                    else:
                        points_right+=1
                if target != 0:
                    if points_left >= points_right: #If more points on the left move left.
                        rob.move_blocking(25,50,250)
                    else:
                        rob.move_blocking(50,25,250)
            else: #If not near, just go backward, most likely in the wall
                rnd = random.random()
                if rnd < 0.5:
                    rob.move_blocking(-25,-50,250)
                else:
                    rob.move_blocking(-50,-25,250)
    # Define the model and set it into train mode, together with the optimizer
    # with torch.autograd.detect_anomaly():
    model = CNNwithLSTM(num_outputs)

    # Eval model in hw
    if not isinstance(rob, SimulationRobobo) or eval_:
        evaluation(rob, model)
        return
    
    # Define optimizer for training
    optimizer = optim.Adam(params=model.parameters(), lr=0.005)
    model = train(rob, model, optimizer)

    torch.save(model.state_dict(), './model.ckpt')

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()




def calc_loss_eval(points, p):
    loss_fn = nn.CrossEntropyLoss()
    target = -1
    width, height, center = 640, 480, 100
    if len(points) > 0:
        points_right, points_left = 0, 0
        for point in points:
            if point[0] >= (width//2-center//2) and point[0] <= (width//2+center//2): #If near center ignore the rest and continue forward
                target = 0
                break
            if point[0] <= (width//2-center//2): #If on the left
                points_left+=1
            else:
                points_right+=1
        if target != 0:
            if points_left >= points_right: #If more points on the left move left.
                target = 2
            else:
                target = 3
    else: #If not near, just go backward, most likely in the wall
        target = 1

    target = torch.tensor(target, dtype=torch.long)
    loss = loss_fn(p, target)
    return loss


def get_xyz(position: Position):
    return str({"x":position.x, "y":position.y, "z":position.z})
