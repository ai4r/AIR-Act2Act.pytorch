# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import glob
import random
import argparse
from datetime import timedelta
import cv2
from utils.openpose import pose_keypoints

from utils.AIR import norm_features, denorm_features
from utils.draw import Artist
from utils.kinect import BodyGameRuntime
from user.classifier import load_model, load_test_data, classify, lstm_input_length
from setting import ACTIONS, LSTM_MODEL_PATH, TEST_PATH, NORM_METHOD, INPUT_DATA_TYPE, B_HANDS
from setting import W_VIDEO, H_VIDEO
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
    parser.add_argument('-m', '--mode', type=str, help='mode to run', choices=['data', 'kinect', 'webcam'], required=True)
    parser.add_argument('-l', '--model', type=int, help='model number', choices=model_numbers, default=26)
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
    if args.mode == 'webcam':
        for _ in test_with_webcam(model):
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
                features = denorm_features(human, NORM_METHOD, INPUT_DATA_TYPE, B_HANDS)
                artist.update([features], ["None"], [f"{f}/{len(human_data)}"], fps=10)
                yield None, None
            for idx, inputs in enumerate(test_dataset.inputs):
                behaviors, behavior_names = classify(model, [inputs])
                cur_features = denorm_features(inputs[-1][1:], NORM_METHOD, INPUT_DATA_TYPE, B_HANDS)
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
                body = null_body_3D()

            # stack joint information
            distance = body[0]['z'] / MAX_DISTANCE
            inputs.append(np.hstack([distance if distance != 0.0 else 1.0,
                                    norm_features(body, NORM_METHOD, INPUT_DATA_TYPE, B_HANDS)]))
            inputs = inputs[-lstm_input_length:]
            if len(inputs) < lstm_input_length:
                continue

            behaviors, behavior_names = classify(model, [inputs])
            cur_features = denorm_features(inputs[-1][1:], NORM_METHOD, INPUT_DATA_TYPE, B_HANDS)
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


def skel_to_AIR(skel):
    # flip horizontal
    skel = [[W_VIDEO - colorX, colorY] for [colorX, colorY] in skel]

    # openpose skeleton to AIR body
    body = list()
    body.append({'colorX': skel[8][0], 'colorY': skel[8][1]})  # 0: SpineBase
    body.append({'colorX': skel[8][0], 'colorY': skel[8][1]})  # 1: SpineMid
    body.append({'colorX': skel[1][0], 'colorY': skel[1][1]})  # 2: Neck
    body.append({'colorX': skel[0][0], 'colorY': skel[0][1]})  # 3: Head
    body.append({'colorX': skel[5][0], 'colorY': skel[5][1]})  # 4: ShoulderLeft
    body.append({'colorX': skel[6][0], 'colorY': skel[6][1]})  # 5: ElbowLeft
    body.append({'colorX': skel[7][0], 'colorY': skel[7][1]})  # 6: WristLeft
    body.append({'colorX': 0.0, 'colorY': 0.0})  # 7: HandLeft
    body.append({'colorX': skel[2][0], 'colorY': skel[2][1]})  # 8: ShoulderRight
    body.append({'colorX': skel[3][0], 'colorY': skel[3][1]})  # 9: ElbowRight
    body.append({'colorX': skel[4][0], 'colorY': skel[4][1]})  # 10: WristRight
    body.append({'colorX': 0.0, 'colorY': 0.0})  # 11: HandRight
    body.append({'colorX': skel[12][0], 'colorY': skel[12][1]})  # 12: HipLeft
    body.append({'colorX': skel[13][0], 'colorY': skel[13][1]})  # 13: KneeLeft
    body.append({'colorX': skel[14][0], 'colorY': skel[14][1]})  # 14: AnkleLeft
    body.append({'colorX': 0.0, 'colorY': 0.0})  # 15: FootLeft
    body.append({'colorX': skel[9][0], 'colorY': skel[9][1]})  # 16: HipRight
    body.append({'colorX': skel[10][0], 'colorY': skel[10][1]})  # 17: KneeRight
    body.append({'colorX': skel[11][0], 'colorY': skel[11][1]})  # 18: AnkleRight
    body.append({'colorX': 0.0, 'colorY': 0.0})  # 19: FootRight
    body.append({'colorX': (skel[2][0] + skel[5][0]) / 2, 'colorY': (skel[2][1] + skel[5][1]) / 2})  # 20: SpineShoulder
    body.append({'colorX': 0.0, 'colorY': 0.0})  # 21: HandTipLeft
    body.append({'colorX': 0.0, 'colorY': 0.0})  # 22: ThumbLeft
    body.append({'colorX': 0.0, 'colorY': 0.0})  # 23: handTipRight
    body.append({'colorX': 0.0, 'colorY': 0.0})  # 24: ThumbRight
    return body


def null_body_3D():
    body = list()
    for i in range(25):
        body.append({'x': 0.0,
                     'y': 0.0,
                     'z': 0.0})
    return body


def null_body_2D():
    body = list()
    for i in range(25):
        body.append({'colorX': 0.0,
                     'colorY': 0.0})
    return body


def time_to_string(sec):
    time_str = str(timedelta(seconds=float(sec)))
    return f"0{time_str[:-4]}"


def test_with_webcam(model):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W_VIDEO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_VIDEO)

    last = time.time()
    inputs = list()

    while True:
        if time.time() - last > 0.1:
            last = time.time()

            # 2d skeleton from video
            ret, frame = cap.read()
            key_points, output_data = pose_keypoints(frame)

            output_data = cv2.flip(output_data, 3)
            cv2.imshow(f'{W_VIDEO}x{H_VIDEO}', output_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # extract skeleton features
            if len(key_points.shape) != 3:
                continue
            user_key_points = key_points[0]
            if not user_key_points[4].any() or not user_key_points[7].any():
                continue

            skel = user_key_points[:, :2]
            body = skel_to_AIR(skel)
            features = norm_features(body, NORM_METHOD, INPUT_DATA_TYPE, B_HANDS)

            # stack joint information
            inputs.append(features)
            inputs = inputs[-lstm_input_length:]
            if len(inputs) < lstm_input_length:
                continue

            behaviors, behavior_names = classify(model, [inputs])
            print("user:", behavior_names[0])
            yield body, behaviors[0]


if __name__ == '__main__':
    main()
