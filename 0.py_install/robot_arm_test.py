#!/usr/bin/env python3
#coding=utf-8
import os
import sys
import time

# Ensure local Arm_Lib package is importable when script is run from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Arm_Lib import Arm_Device

# Get the robotic arm object
Arm = Arm_Device()
time.sleep(.1)


def main():
    while True:
        Arm.Arm_RGB_set(50, 0, 0)  # RGB lights up red
        time.sleep(.5)
        Arm.Arm_RGB_set(0, 50, 0)  # RGB lights up green
        time.sleep(.5)
        Arm.Arm_RGB_set(0, 0, 50)  # RGB lights up blue
        time.sleep(.5)
        print(" END OF LINE! ")


try:
    # main()
    Arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 120, 1000)
except KeyboardInterrupt:
    # Release Arm object
    del Arm
    print(" Program closed! ")
    pass