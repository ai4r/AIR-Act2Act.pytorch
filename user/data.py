# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from tqdm import tqdm

from torch.utils import data

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.AIR import norm_features
from user.constants import KINECT_FRAME_RATE, TARGET_FRAME_RATE
from user.k_clustering import load_proper_model
from setting import gen_sequence, ACTIONS, NORM_METHOD


class AIRDataSet(data.Dataset):
    def __init__(self, data_path, dim_input, dim_output, data_name=None, b_add_noise=False, b_connect_sequence=False):
        # load data from files
        self.data_path = data_path
        self.data_name = data_name
        self.file_names = list()
        if data_name is not None:
            self.file_names.append(os.path.join(self.data_path, self.data_name))
        else:
            for action in ACTIONS:
                self.file_names.extend(glob.glob(os.path.join(self.data_path, f"*{action}*.npz")))
        path = data_path if data_name is None else os.path.join(data_path, data_name)
        print(f'Data loading... ({path})')
        print(f'Total {len(self.file_names)} files.')

        self.human_data = list()
        self.robot_data = list()
        self.third_data = list()
        step = round(KINECT_FRAME_RATE / TARGET_FRAME_RATE)
        for file in self.file_names:
            with np.load(file, allow_pickle=True) as data:
                self.human_data.append([norm_features(human, NORM_METHOD) for human in data['human_info']][::step])
                self.robot_data.append([norm_features(robot, NORM_METHOD) for robot in data['robot_info']][::step])
                self.third_data.append(data['third_info'][::step])

        # extract training data
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.b_add_noise = b_add_noise
        self.b_connect_sequence = b_connect_sequence

        self.inputs = list()
        self.outputs = list()
        seq_length = self.dim_input[0]
        pbar = tqdm(total=len(self.third_data))
        for idx, third in enumerate(self.third_data):
            if all(v == 1.0 for v in third):
                continue
            for human_seq, third_seq in zip(gen_sequence(self.human_data[idx], seq_length),
                                            gen_sequence(self.third_data[idx], seq_length)):
                seq = np.concatenate((third_seq, human_seq), axis=1)
                self.inputs.append(seq)
                action_name = [action for action in ACTIONS if action in self.file_names[idx]][0]
                cur_action = self.recog_subaction(seq, action_name)
                self.outputs.append(cur_action)

            pbar.update(1)
        pbar.close()
        # self.inputs = np.round(self.inputs, 3)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item].astype("float32"), self.outputs[item]

    def recog_subaction(self, sequence, action):
        kmeans, sub_action_mapping = load_proper_model(action)
        df = kmeans.make_dataframe([sequence], len(sequence))
        sub_action = kmeans.km_model.predict(df)[0]
        return sub_action_mapping[sub_action]

    def add_random_noise(self):
        pass

    def connect_sequence(self):
        pass
