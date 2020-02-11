# -*- coding: utf-8 -*-
# 1단계: 사용자 상위행위 인식 테스트
# 2단계: 사용자 하위행위 인식 테스트
# 3단계: 상호작용 행위 생성 테스트 (matplotlib으로 3차원 디스플레이)
import torch

import os
import glob
import random
import numpy as np

from model import Act2Act
from data import ACTIONS, KINECT_FRAME_RATE, TARGET_FRAME_RATE, gen_sequence
from utils.AIR import norm_to_torso

# define LSTM model
lstm_input_length = 30
lstm_input_size = 25
hidden_size = 1024
output_dim = len(ACTIONS)
model = Act2Act(lstm_input_size, hidden_size, output_dim)
model.cuda()
model.load_state_dict(torch.load("./models/model_0900.pth"))

# load test data
test_path = './data files/'
data_files = list()
for action in ACTIONS:
    data_files.extend(glob.glob(os.path.join(test_path, f"*{action}*.npz")))
random.shuffle(data_files)
print(f'There are {len(data_files)} data.')
for idx in range(min(len(data_files), 20)):
    data_file = os.path.basename(data_files[idx])
    data_name, _ = os.path.splitext(data_file)
    print(f'{idx}: {data_name}')

# select data name to test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
while True:
    var = int(input("Input data number to display: "))
    test_file = data_files[var]
    print(os.path.normpath(test_file))

    with np.load(test_file, allow_pickle=True) as data:
        human_data = [norm_to_torso(human) for human in data['human_info']]
        third_data = data['third_info']
        step = round(KINECT_FRAME_RATE / TARGET_FRAME_RATE)
        sampled_human_data = human_data[::step]
        sampled_third_data = third_data[::step]

        # ground truth
        action_name = [action for action in ACTIONS if action in test_file][0]
        cur_action = ACTIONS.index(action_name)

        # prediction results
        outputs = list()
        predictions = list()
        for human_seq, third_seq in zip(gen_sequence(sampled_human_data, lstm_input_length),
                                        gen_sequence(sampled_third_data, lstm_input_length)):
            inputs = np.concatenate((third_seq, human_seq), axis=1)
            input_batch = torch.FloatTensor([inputs]).to(device)
            scores = model(input_batch)
            prediction = torch.argmax(scores, dim=1)
            predictions.append(prediction.item())
            outputs.append(cur_action)

        print("true: ", outputs)
        print("pred: ", predictions)
