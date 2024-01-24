import time
import cv2
import numpy as np

from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
)


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())

def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.stop_simulation()
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.play_simulation()
    print(rob.get_sim_time())
    print(rob.position())


def run_all_actions(rob: IRobobo):

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    test_emotions(rob)
    test_sensors(rob)
    test_move_and_wheel_reset(rob)
    if isinstance(rob, SimulationRobobo):
        test_sim(rob)
        rob.set_realtime()

    test_phone_movement(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def test_sens(rob: IRobobo):
    print('\t'.join(['BackL', 'BackR', 'FrontL', 'FrontR', 'FrontC', 'FrontRR', 'BackC', 'FrontLL']))

    while True:
        irs = rob.read_irs()
        print('\t'.join([str(x) for x in irs]))
    
def run_obstacle_avoidance(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    obstacle_threshold = 100

    while True:
        irs = rob.read_irs()
        print('\t'.join([str(x) for x in irs]))

        # Check if any infrared sensor detects an obstacle
        irs2 = irs[1:6] + [irs[7]]
        if any(val > obstacle_threshold for val in irs2):
            print("Obstacle detected! Adjusting direction...")
            left_sensors = [irs[7], irs[2], irs[4]]
            right_sensors = [irs[4], irs[3], irs[5]]
            # calculate the difference in sensor readings for left and right sides
            left_side = sum(left_sensors) / 3  # average left side sensors
            #print(f'left side = {left_side}')

            right_side = sum(right_sensors) / 3  # average right side sensors
            #print(f'right side = {right_side}')

            difference = left_side - right_side

            # calculate wheel speeds for turning
            left_speed = int(50 + difference)  
            #print(f'left speed = {left_speed}')
            right_speed = int(50 - difference) 
            #print(f'right speed = {right_speed}')

            # keep speeds within valid range
            left_speed = max(-100, min(100, left_speed))
            right_speed = max(-100, min(100, right_speed))

            # for debugging purpose
            # print(f"Adjusted Speeds: Left={left_speed}, Right={right_speed}")
            print('going to make a turn')

            # move the robot with adjusted wheel speeds for turning
            rob.move_blocking(left_speed, right_speed, 200)
            #rob.move_blocking(100, 100, 500)  # adjust duration for a longer turn
            time.sleep(0.25)

        else:
            # if no obstacle detected, move forward
            rob.move_blocking(90, 90, 125)
            time.sleep(0.25)
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

def run_to_block_and_stop(rob: IRobobo):
    print('\t'.join(['BackL', 'BackR', 'FrontL', 'FrontR', 'FrontC', 'FrontRR', 'BackC', 'FrontLL']))
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    thresh = 1000

    while True:
        irs = rob.read_irs()
        print('\t'.join([str(x) for x in irs]))
        if irs[2] > thresh or irs[3] > thresh or irs[4] > thresh or irs[5] > thresh or irs[7] > thresh:
            break
        rob.move_blocking(50, 50, 125)
        rob.sleep(0.25)

    # for i in range(50):
    #     # print('\t'.join([str(x) for x in rob.read_irs()]))
    #     print(rob.read_irs())
    #     rob.move_blocking(50, 50, 125)
    #     rob.sleep(0.25)
    rob.sleep(20)
    rob.reset_wheels()
    rob.sleep(1)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        
def forward_backward(rob: IRobobo):
    print('\t'.join(['BackL', 'BackR', 'FrontL', 'FrontR', 'FrontC', 'FrontRR', 'BackC', 'FrontLL']))
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    thresh = 10000
    forward = True
    
    while True:
        irs = rob.read_irs()
        print('\t'.join([str(x) for x in irs]))
        if forward:
            if max(irs[2], irs[3], irs[4], irs[5], irs[7]) > thresh:
                forward = False
            rob.move_blocking(50, 50, 125)
            rob.sleep(0.25)
        else:
            if max(irs[0], irs[1], irs[6]) > thresh:
                break
            rob.move_blocking(-50, -50, 125)
            rob.sleep(0.25)

    # for i in range(50):
    #     # print('\t'.join([str(x) for x in rob.read_irs()]))
    #     print(rob.read_irs())
    #     rob.move_blocking(50, 50, 125)
    #     rob.sleep(0.25)
    rob.sleep(20)
    rob.reset_wheels()
    rob.sleep(1)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

def blob_detection(rob: IRobobo):
    #TODO: maybe change the camera size
    camera_width = 640
    camera_height = 480

    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 0  # 0 for dark blobs, 255 for light blobs

    #circularity for rectangles
    params.filterByCircularity = True
    params.minCircularity = 0.6 

    #convexity completely covered
    params.filterByConvexity = True
    params.minConvexity = 0.9  

    #inertia ratio (for rectangles)
    params.filterByInertia = True
    params.minInertiaRatio = 0.6 


    detector = cv2.SimpleBlobDetector_create(params)

    while True:
        frame = rob.get_image_front()

        #TODO:resize if needed
        frame = cv2.resize(frame, (camera_width, camera_height))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        #mask for the green color
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        #AND operation to get the green regions
        green_regions = cv2.bitwise_and(frame, frame, mask=green_mask)

        gray_frame = cv2.cvtColor(green_regions, cv2.COLOR_BGR2GRAY)

        #perform blob detection
        keypoints = detector.detect(gray_frame)

        if keypoints:
            keypoint = keypoints[0]
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            size_percent = (keypoint.size / (camera_width * camera_height)) * 100

            #x and y values along with the percentage of blob area
            return x, y, size_percent

        time.sleep(3)