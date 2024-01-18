import cv2
import time

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

        print("Test")
        time.sleep(2)
        print("Restarting")
        rob.play_simulation()
    
    
def test_environment(rob: IRobobo):
    rob.play_simulation()
    print("=============================")
    print(dir(rob))
    
    #get_info(rob)
    #rob.move_blocking(50, 50, 125)

    target_position = rob.get_target_position()
    current_position = rob.position()

    angle_xz = math.atan2(delta_z, delta_x)
    angle_xz_degrees = math.degrees(angle_xz)

    print(target_position)
    print(current_position)
    #get_info(rob)

    rob.stop_simulation()

def get_info(rob: IRobobo):
    print(rob.position())
    print(rob.read_accel())
    print(rob.read_orientation())
    print(rob.read_wheels())