#!/bin/bash
conda activate LM2023 &
coppeliaSim -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE ./scenes/Robobo_Scene.ttt

sh ./script/run.sh