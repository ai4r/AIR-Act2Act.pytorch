# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from tqdm import tqdm

from torch.utils import data

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.AIR import norm_features
from user.k_clustering import KMeansClustering
from user.constants import KINECT_FRAME_RATE, TARGET_FRAME_RATE
from user.constants import sub_action_mapping_1, sub_action_mapping_2, sub_action_mapping_3
from setting import gen_sequence, NORM_METHOD, ACTIONS, TRAIN_PATH, TEST_PATH, K_MEANS_MODEL_PATH


class AIRDataSet(data.Dataset):
    def __init__(self, data_path, dim_input, dim_output, data_name=None, b_add_noise=False, b_connect_sequence=False):
        # load k-means clustering models
        seq_length = dim_input[0]
        train_paths = [TRAIN_PATH, TEST_PATH]
        model_path = K_MEANS_MODEL_PATH
        norm_method = NORM_METHOD

        actions_1 = ["A001", "A004"]
        n_clusters_1 = 4
        self.kmeans_1 = KMeansClustering(actions_1, n_clusters_1, seq_length, norm_method, train_paths, model_path)
        actions_2 = ["A004", "A005", "A006"]
        n_clusters_2 = 9
        self.kmeans_2 = KMeansClustering(actions_2, n_clusters_2, seq_length, norm_method, train_paths, model_path)
        actions_3 = ["A004", "A005", "A008"]
        n_clusters_3 = 7
        self.kmeans_3 = KMeansClustering(actions_3, n_clusters_3, seq_length, norm_method, train_paths, model_path)
        # print('K-means models are loaded.')

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
                self.human_data.append([norm_features(human, norm_method) for human in data['human_info']][::step])
                self.robot_data.append([norm_features(robot, norm_method) for robot in data['robot_info']][::step])
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
        if action == 'A001' or action == 'A004':
            kmeans = self.kmeans_1
            sub_action_mapping = sub_action_mapping_1
        elif action == 'A005' or action == 'A006':
            kmeans = self.kmeans_2
            sub_action_mapping = sub_action_mapping_2
        elif action == 'A008':
            kmeans = self.kmeans_3
            sub_action_mapping = sub_action_mapping_3
        else:
            raise Exception(f'Wrong action name: {action}')

        df = kmeans.make_dataframe([sequence], len(sequence))
        sub_action = kmeans.km_model.predict(df)[0]
        return sub_action_mapping[sub_action]

    def add_random_noise(self):
        pass

    def connect_sequence(self):
        pass
