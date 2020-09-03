# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import glob
import random
import argparse
from datetime import timedelta

from utils.AIR import norm_features, denorm_features
from utils.draw import Artist
from utils.kinect import BodyGameRuntime
from user.classifier import load_model, load_test_data, classify, lstm_input_length
from setting import ACTIONS, LSTM_MODEL_PATH, TEST_PATH, NORM_METHOD
from preprocess import MAX_DISTANCE


# gather all existing models
model_files = glob.glob(os.path.join(LSTM_MODEL_PATH, "*.pth"))
model_numbers = list()
for model_file in model_files:
    model_name, _ = os.path.splitext(os.path.basename(model_file))
    model_numbers.append(int(model_name[6:10]))


# argument parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='mode to run', choices=['data', 'kinect'], required=True)
    parser.add_argument('-l', '--model', type=int, help='model number', choices=model_numbers, default=13)
    args = parser.parse_args()

    MODEL_FILE = os.path.join(LSTM_MODEL_PATH, f"model_{args.model:04d}.pth")
    model = load_model(MODEL_FILE)


def main():
    if args.mode == "data":
        for _ in test_with_data(model):
            pass
    if args.mode == "kinect":
        for _ in test_with_kinect(model):
            pass


def test_with_data(model):
    # load test data
    data_files = list()
    for action in ACTIONS:
        data_files.extend(glob.glob(os.path.join(TEST_PATH, f"*{action}*.npz")))
    print(f'\nThere are {len(data_files)} data.')
    random.shuffle(data_files)

    # select data to test and return input and output
    artist = Artist(n_plot=1)
    while True:
        for idx in range(min(len(data_files), 20)):
            data_file = os.path.basename(data_files[idx])
            data_name, _ = os.path.splitext(data_file)
            print(f'{idx}: {data_name}')

        var = int(input("Input data number to display: "))
        test_file = data_files[var]
        print(os.path.normpath(test_file))

        if os.path.exists(test_file):
            test_dataset = load_test_data(data_name=test_file)
            human_data = test_dataset.human_data[0]
            for f, human in enumerate(human_data[:test_dataset.dim_input[0]]):
                features = denorm_features(human, NORM_METHOD)
                artist.update([features], ["None"], [f"{f}/{len(human_data)}"], fps=10)
                yield None, None
            for idx, inputs in enumerate(test_dataset.inputs):
                behaviors, behavior_names = classify(model, [inputs])
                cur_features = denorm_features(inputs[-1][1:], NORM_METHOD)
                f += 1
                artist.update([cur_features], [behavior_names[0]], [f"{f}/{len(human_data)}"], fps=10)
                yield None, behaviors[0]
        else:
            print("Wrong data number.")


def test_with_kinect(model):
    game = BodyGameRuntime()
    last = time.time()
    first = True
    inputs = list()

    artist = Artist(n_plot=1)
    start_time = time.time()
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

            # stack joint information
            distance = body[0]['z'] / MAX_DISTANCE
            inputs.append(np.hstack([distance if distance != 0.0 else 1.0,
                                    norm_features(body, NORM_METHOD)]))
            inputs = inputs[-lstm_input_length:]
            if len(inputs) < lstm_input_length:
                pass

            behaviors, behavior_names = classify(model, [inputs])
            cur_features = denorm_features(inputs[-1][1:], NORM_METHOD)
            elapsed_time = time_to_string(time.time() - start_time)
            artist.update([cur_features], [behavior_names[0]], [elapsed_time])
            yield body, behaviors[0]


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


def time_to_string(sec):
    time_str = str(timedelta(seconds=float(sec)))
    return f"0{time_str[:-4]}"


if __name__ == '__main__':
    main()
