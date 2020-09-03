# -*- coding: utf-8 -*-
import os
import glob
import time
import socket
import simplejson as json
import argparse

from recognize import test_with_data, test_with_kinect, load_model
from robot.adapter import adapt_behavior
from robot.selector import select_behavior
from setting import LSTM_MODEL_PATH

# gather all existing models
model_files = glob.glob(os.path.join(LSTM_MODEL_PATH, "*.pth"))
model_numbers = list()
for model_file in model_files:
    model_name, _ = os.path.splitext(os.path.basename(model_file))
    model_numbers.append(int(model_name[6:10]))

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--use_robot', help='use robot', action='store_true')
parser.add_argument('-m', '--mode', type=str, help='mode to run', choices=['data', 'kinect'], required=True)
parser.add_argument('-l', '--model', type=int, help='model number', choices=model_numbers, default=13)
args = parser.parse_args()

MODEL_FILE = os.path.join(LSTM_MODEL_PATH, f"model_{args.model:04d}.pth")
model = load_model(MODEL_FILE)


def main():
    # create socket connection
    if args.use_robot:
        HOST = "127.0.0.1"
        CMD_PORT = 10240
        cmd_sock = init_socket(HOST, CMD_PORT)
        send_behavior(cmd_sock, adapt_behavior('stand'))
    else:
        cmd_sock = None

    # user behavior recognition
    if args.mode == "data":
        input_generator = test_with_data(model)
    if args.mode == "kinect":
        input_generator = test_with_kinect(model)

    # select and adapt behavior
    user_behaviors = list()
    for user_pose, user_behavior in input_generator:
        if user_behavior is None:
            continue

        user_behaviors.append(user_behavior)
        user_behaviors = user_behaviors[-3:]
        if len(user_behaviors) < 3:
            continue

        robot_behavior = select_behavior(user_behaviors)
        if robot_behavior is None:
            continue

        print(robot_behavior)
        if args.use_robot:
            robot_pose = adapt_behavior(robot_behavior, user_pose)
            send_behavior(cmd_sock, robot_pose)


# wait for connection to server
def init_socket(HOST, CMD_PORT):
    while True:
        try:
            cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            cmd_sock.connect((HOST, CMD_PORT))
            print("Connect to %s" % str(HOST))
            return cmd_sock
        except socket.error:
            print("Connection failed, retrying...")
            time.sleep(3)


# send pose to server
def send_behavior(cmd_sock, pose):
    json_string = json.dumps({'target_angles': pose})
    cmd_sock.send(str(len(json_string)).ljust(16).encode('utf-8'))
    cmd_sock.sendall(json_string.encode('utf-8'))


if __name__ == '__main__':
    main()
