from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
)

from .lstm import eucl_loss_fn


def run_lstm_sim(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    for _ in range(50):
        target_pos = rob.get_target_position()
        rob_pos = rob.position()
        print(eucl_loss_fn((rob_pos.x, rob_pos.y), (target_pos.x, target_pos.y)))
        rob.move_blocking(50, 50, 250)
        rob.sleep(0.250)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()