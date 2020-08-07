# -*- coding: utf-8 -*-
import os
import glob
import time
import socket
import simplejson as json
import argparse
from textwrap import wrap

import torch
from model import Act2Act
from data import AIRDataSet, ACTIONS, norm_method
from utils.AIR import denorm_features
from utils.draw import animation, plt, init_axis, draw_parts
from constants import SUBACTION_NAMES
from constants import POSE_IDS, POSES


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
parser.add_argument('-m', '--mode', type=str, help='mode to run', choices=['test', 'test_pepper'], required=True)
args = parser.parse_args()

# global variable
MODEL_FILE = os.path.join(MODEL_PATH, f"model_{args.model:04d}.pth")
TEST_PATH = './data files/valid data'


def test():
    # connect to server
    if args.mode == "test_pepper":
        HOST = "127.0.0.1"
        CMD_PORT = 10240
        cmd_sock = init_socket(HOST, CMD_PORT)
        send_behavior(cmd_sock, 'stand')
    else:
        cmd_sock = None

    # define model parameters
    lstm_input_length = 15
    lstm_input_size = 25
    hidden_size = 1024
    output_dim = len(SUBACTION_NAMES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define LSTM model
    model = Act2Act(device, lstm_input_size, hidden_size, output_dim)
    model.to(device)
    print("Load model: ", MODEL_FILE)
    model.load_state_dict(torch.load(MODEL_FILE))

    # load test data
    data_files = list()
    for action in ACTIONS:
        data_files.extend(glob.glob(os.path.join(TEST_PATH, f"*{action}*.npz")))
    # random.shuffle(data_files)
    data_files.sort()

    # select data to test
    while True:
        print(f'There are {len(data_files)} data.')
        for idx in range(min(len(data_files), 50)):
            data_file = os.path.basename(data_files[idx])
            data_name, _ = os.path.splitext(data_file)
            print(f'{idx}: {data_name}')

        var = int(input("Input data number to display: "))
        test_file = data_files[var]
        print(os.path.normpath(test_file))

        test_dataset = AIRDataSet(data_path=TEST_PATH,
                                  data_name=os.path.basename(test_file),
                                  dim_input=(lstm_input_length, lstm_input_size),
                                  dim_output=(output_dim, 1))

        # prediction results
        outputs = list()
        predictions = list()
        for idx, inputs in enumerate(test_dataset.inputs):
            input_batch = torch.FloatTensor([inputs]).to(device)
            scores = model(input_batch)
            prediction = torch.argmax(scores, dim=1)
            predictions.append(prediction.item())
            outputs.append(test_dataset.outputs[idx])
        print("true: \n", outputs)
        print("pred: \n", predictions)

        # draw results
        features = list()
        for f in range(len(test_dataset.human_data[0])):
            cur_features = test_dataset.human_data[0][f]
            cur_features = denorm_features(cur_features, norm_method)
            features.append(cur_features)
        predictions = ["None"] * (lstm_input_length - 1) + predictions

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        anim = animation.FuncAnimation(fig, animate_3d, fargs=(features, predictions, ax, cmd_sock),
                                       frames=len(features), interval=200, blit=True, repeat=True)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        plt.close()


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


def animate_3d(f, features, results, ax, cmd_sock):
    ret_artists = list()

    init_axis(ax)
    pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist = features[f]
    ret_artists.extend(draw_parts(ax, [pelvis, neck, head]))
    ret_artists.extend(draw_parts(ax, [neck, lshoulder, lelbow, lwrist]))
    ret_artists.extend(draw_parts(ax, [neck, rshoulder, relbow, rwrist]))

    result = "None" if results[f] == "None" else "\n".join(wrap(SUBACTION_NAMES[results[f]], 15))
    ret_artists.append(ax.text(0, 0, 0, F"{result}\n{f+1}/{len(features)}", fontsize=20))

    # send behavior to client
    if f == 0 or f == 1:
        send_behavior(cmd_sock, 'stand')
    else:
        results = results[f-2:f+1]
        if results == [0, 0, 0] or results == [6, 6, 6] or results == [10, 10, 10]:
            send_behavior(cmd_sock, 'stand')
        elif results == [1, 1, 1]:
            send_behavior(cmd_sock, 'bow')
        elif results == [4, 4, 4]:
            send_behavior(cmd_sock, 'handshake')
        elif results == [7, 7, 7] or results == [9, 9, 9] or results == [11, 11, 11]:
            send_behavior(cmd_sock, 'hug')
        elif results == [12, 12, 12] or results == [13, 13, 13]:
            send_behavior(cmd_sock, 'avoid')

    return ret_artists


def send_behavior(cmd_sock, behavior):
    print(behavior)
    if args.mode == "test_pepper":
        json_string = json.dumps({'target_angles': POSES[POSE_IDS[behavior]]})
        cmd_sock.send(str(len(json_string)).ljust(16).encode('utf-8'))
        cmd_sock.sendall(json_string.encode('utf-8'))


def main():
    print("Model path: ", MODEL_FILE)
    if not os.path.exists(MODEL_FILE):
        raise Exception("Cannot load the model.")
    test()

if __name__ == '__main__':
    main()
