# -*- coding: utf-8 -*-
import os
import time
import socket
import numpy as np
import simplejson as json
import glob
import argparse

import torch
from model import Act2Act
from data import norm_method
from utils.AIR import norm_features
from constants import SUBACTION_NAMES
from constants import POSE_IDS, POSES

from utils.kinect import BodyGameRuntime
from preprocess import MAX_DISTANCE


# gather all existing models
MODEL_PATH = './models/lstm/vector'
model_files = glob.glob(os.path.join(MODEL_PATH, "*.pth"))
model_numbers = list()
for model_file in model_files:
    model_name, _ = os.path.splitext(os.path.basename(model_file))
    model_numbers.append(int(model_name[6:10]))

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--model', type=int, help='model number', choices=model_numbers, default=13)
parser.add_argument('-m', '--mode', type=str, help='mode to run', choices=['recognize', 'generate'], required=True)
args = parser.parse_args()

# global variable
MODEL_FILE = os.path.join(MODEL_PATH, f"model_{args.model:04d}.pth")


def pose_to_AIR(pose):
    body = list()
    for i in range(25):
        body.append({'x': pose[i].Position.x,
                     'y': pose[i].Position.y,
                     'z': pose[i].Position.z})
    return body


def null_features():
    body = list()
    for i in range(25):
        body.append({'x': 0.0,
                     'y': 0.0,
                     'z': 0.0})
    return body


def run_kinect():
    # connect to server
    if args.mode == "generate":
        HOST = "127.0.0.1"
        CMD_PORT = 10240
        cmd_sock = init_socket(HOST, CMD_PORT)
        send_behavior(cmd_sock, 'stand')

    # define model parameters
    lstm_input_length = 15
    lstm_input_size = 25
    hidden_size = 1024
    output_dim = len(SUBACTION_NAMES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load LSTM model
    model = Act2Act(device, lstm_input_size, hidden_size, output_dim)
    model.to(device)
    model.load_state_dict(torch.load(MODEL_FILE))

    # run KINECT camera
    game = BodyGameRuntime()
    last = time.time()
    first = True
    inputs = list()
    predictions = list()
    for joints in game.run():
        if time.time() - last > 0.1 and (len(joints) > 0 or not first):
            first = False
            last = time.time()
            if len(joints) > 0:
                depths = [joint[20].Position.z for joint in joints]  # 20: spineShoulder
                idx = depths.index(min(depths))
                body = pose_to_AIR(joints[idx])
            else:
                body = null_features()
            # print(body)

            # prediction
            distance = body[0]['z'] / MAX_DISTANCE
            inputs.append(np.hstack([distance if distance != 0.0 else 1.0,
                                    norm_features(body, norm_method)]))
            inputs = inputs[-lstm_input_length:]
            if len(inputs) < lstm_input_length:
                pass

            input_batch = torch.FloatTensor([inputs]).to(device)
            scores = model(input_batch)
            prediction = torch.argmax(scores, dim=1)
            print(SUBACTION_NAMES[prediction.item()])

            # send behavior to Pepper robot
            if args.mode == "generate":
                predictions.append(prediction.item())
                predictions = predictions[-3:]
                if len(predictions) == 3:
                    # print(predictions)
                    if predictions == [0, 0, 0] or predictions == [6, 6, 6] or predictions == [10, 10, 10]:
                        send_behavior(cmd_sock, 'stand', body)
                    elif predictions == [3, 3, 1] or predictions == [3, 3, 0]:
                        send_behavior(cmd_sock, 'bow', body)
                    elif predictions == [4, 4, 4]:
                        send_behavior(cmd_sock, 'handshake', body)
                    elif predictions == [7, 7, 7] or predictions == [9, 9, 9] or predictions == [11, 11, 11]:
                        send_behavior(cmd_sock, 'hug', body)
                    elif predictions == [12, 12, 12] or predictions == [13, 13, 13]:
                        send_behavior(cmd_sock, 'avoid', body)


def init_socket(HOST, CMD_PORT):
    # connect to server
    while True:
        try:
            cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            cmd_sock.connect((HOST, CMD_PORT))
            print("Connect to %s" % str(HOST))
            return cmd_sock
        except socket.error:
            print("Connection failed, retrying...")
            time.sleep(3)
    print("Server connected")


def send_behavior(cmd_sock, behavior, body=None):
    # print(behavior)
    if behavior == "hug":
        shoulder_pos_y = body[20]['y']
        shoulder_pos_z = body[20]['z']
        angles = hug_behavior(POSES[POSE_IDS["stand"]], POSES[POSE_IDS["hug"]], shoulder_pos_y, shoulder_pos_z)
    elif behavior == "handshake":
        hand_pos_x = body[10]['x']  #wristRight
        hand_pos_y = body[10]['y']
        angles = handshake_behavior(POSES[POSE_IDS["handshake"]], hand_pos_x, hand_pos_y)
    else:
        angles = POSES[POSE_IDS[behavior]]

    json_string = json.dumps({'target_angles': angles})
    cmd_sock.send(str(len(json_string)).ljust(16).encode('utf-8'))
    cmd_sock.sendall(json_string.encode('utf-8'))


# 사용자 어깨 높이를 반영하여 hug behavior 생성
def hug_behavior(stand_pose, hug_pose, shoulder_pos_y, shoulder_pos_z):
    head_angle = 20
    if shoulder_pos_y > -0.7:
        diff = [a - b for a, b in zip(hug_pose, stand_pose)]
        final_pose = [a + (shoulder_pos_y + 0.7) * b for a, b in zip(stand_pose, diff)]
    else:
        final_pose = stand_pose
    return final_pose


# 사용자 어깨 높이를 반영하여 hug behavior 생성
def hug_behavior(pose, ):
    pass


# 사용자 손 위치를 반영하여 handshake behavior 생성
def handshake_behavior():
    pass


def main():
    print("Model path: ", MODEL_FILE)
    if not os.path.exists(MODEL_FILE):
        raise Exception("Cannot load the model.")
    run_kinect()

if __name__ == '__main__':
    main()
