# -*- coding: utf-8 -*-
# 1단계: 사용자 상위행위 인식 테스트
# 2단계: 사용자 하위행위 인식 테스트
# 3단계: 상호작용 행위 생성 테스트 (matplotlib으로 3차원 디스플레이)
import torch

import os
import glob
import random
import numpy as np

from model import Act2Act, Encoder, Decoder
from data import ACTIONS, KINECT_FRAME_RATE, TARGET_FRAME_RATE, gen_sequence
from utils.AIR import norm_to_torso, denorm_from_torso
from utils.robot import norm_to_joint_angles, denorm_from_joint_angles
from animate import draw


# define Act2Act model
lstm_input_length = 20
lstm_input_size = 25
lstm_output_length = 5
lstm_output_size = 10
hidden_size = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(lstm_input_size, hidden_size, device, n_layers=1, dropout=.0)
decoder = Decoder(lstm_output_size, hidden_size, n_layers=1, dropout=.0)
model = Act2Act(encoder, decoder, device).to(device)

# show all existing models
MODEL_PATH = './models/'
model_files = glob.glob(os.path.join(MODEL_PATH, "*.pth"))
model_names = list()
for model_file in model_files:
    model_name, _ = os.path.splitext(os.path.basename(model_file))
    model_names.append(model_name)
print(f'There are {len(model_files)} models ({model_names[0]} ~ {model_names[-1]}).')

# select model to test
while True:
    model_num = input(f"Input model number to test ({model_names[0][6:]} ~ {model_names[-1][6:]}): ")
    model_num = int(model_num)
    selected_model = os.path.join(MODEL_PATH, f"model_{model_num:04d}.pth")
    if os.path.exists(selected_model):
        print("Load model: ", selected_model)
        model.load_state_dict(torch.load(selected_model))
        break
    print("Model number is wrong.")

# load test data
TEST_PATH = './data files/valid data/'
data_files = list()
for action in ACTIONS:
    data_files.extend(glob.glob(os.path.join(TEST_PATH, f"*{action}*.npz")))
random.shuffle(data_files)
print(f'There are {len(data_files)} data.')
for idx in range(min(len(data_files), 20)):
    data_file = os.path.basename(data_files[idx])
    data_name, _ = os.path.splitext(data_file)
    print(f'{idx}: {data_name}')

# select data to test
while True:
    var = int(input("Input data number to display: "))
    test_file = data_files[var]
    print(os.path.normpath(test_file))

    with np.load(test_file, allow_pickle=True) as data:
        with torch.no_grad():
            # handle data
            human_data = [norm_to_torso(human) for human in data['human_info']]
            robot_data = [norm_to_joint_angles(robot) for robot in data['robot_info']]
            third_data = data['third_info']

            step = round(KINECT_FRAME_RATE / TARGET_FRAME_RATE)
            sampled_human_data = human_data[::step]
            sampled_robot_data = robot_data[::step]
            sampled_third_data = third_data[::step]

            # ground truth
            outputs = sampled_robot_data

            # prediction
            predictions = sampled_robot_data[:lstm_input_length]
            for human_seq, third_seq in zip(gen_sequence(sampled_human_data, lstm_input_length),
                                            gen_sequence(sampled_third_data, lstm_input_length)):
                encoder_inputs = np.array([np.concatenate((third_seq, human_seq), axis=1)], dtype='float32')
                decoder_input = np.array([predictions[-1:]], dtype='float32')
                decoder_outputs = np.array([predictions[-1:]], dtype='float32')
                prediction = model(torch.from_numpy(encoder_inputs).to(device),
                                   torch.from_numpy(decoder_input).to(device),
                                   torch.from_numpy(decoder_outputs).to(device),
                                   teacher_forcing_ratio=.0)
                predictions.append(prediction.cpu().data.numpy()[0][0])

            # draw results
            outputs = [denorm_from_joint_angles(output) for output in outputs]
            predictions = [denorm_from_joint_angles(prediction) for prediction in predictions]
            draw([outputs, predictions], save_path=None, b_show=True)
