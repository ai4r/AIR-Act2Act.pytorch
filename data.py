# -*- coding: utf-8 -*-
# AIR-Act2Act 데이터셋을 불러와서 data augmentation 하기
# data augmentation 1: skeleton 정보에 random noise 추가
# data augmentation 2: action 간 부드럽게 연결된 데이터 생성
from torch.utils import data

import os
import glob
import numpy as np

from utils.AIR import norm_to_torso

KINECT_FRAME_RATE = 30  # frame rate of kinect camera
TARGET_FRAME_RATE = 10  # frame rate of extracted data
ACTIONS = ["A%03d" % a for a in range(1, 11)]
# ACTIONS = ['A001', 'A005']


def gen_sequence(data, length):
    for start_idx in range(len(data) - length + 1):
        yield list(data[start_idx:start_idx + length])


class AIRDataSet(data.Dataset):
    def __init__(self, data_path, dim_input, dim_output, b_add_noise=False, b_connect_sequence=False):
        # load data from files
        self.data_path = data_path
        self.file_names = list()
        for action in ACTIONS:
            self.file_names.extend(glob.glob(os.path.join(self.data_path, f"*{action}*.npz")))
        print(f'Data loading. Total {len(self.file_names)} files.')

        self.human_data = list()
        self.robot_data = list()
        self.third_data = list()
        for file in self.file_names:
            with np.load(file, allow_pickle=True) as data:
                self.human_data.append([norm_to_torso(human) for human in data['human_info']])
                self.robot_data.append([norm_to_torso(robot) for robot in data['robot_info']])
                self.third_data.append(data['third_info'])

        # extract training data
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.b_add_noise = b_add_noise
        self.b_connect_sequence = b_connect_sequence

        self.inputs = list()
        self.outputs = list()
        step = round(KINECT_FRAME_RATE / TARGET_FRAME_RATE)
        seq_length = self.dim_input[0]
        for idx, third in enumerate(self.third_data):
            if all(v == 1.0 for v in third):
                continue

            sampled_human_seq = self.human_data[idx][::step]
            sampled_third_seq = self.third_data[idx][::step]
            for human_seq, third_seq in zip(gen_sequence(sampled_human_seq, seq_length),
                                            gen_sequence(sampled_third_seq, seq_length)):
                self.inputs.append(np.concatenate((third_seq, human_seq), axis=1))
                action_name = [action for action in ACTIONS if action in self.file_names[idx]][0]
                cur_action = ACTIONS.index(action_name)
                self.outputs.append(cur_action)

            # sampled_robot_seq = self.robot_data[idx][::step]  # 나중에 사용할 예정
        # self.inputs = np.round(self.inputs, 3)
        print('Data extracted.')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item].astype("float32"), self.outputs[item]

    def add_random_noise(self):
        pass

    def connect_sequence(self):
        pass
