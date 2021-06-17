import os
import glob
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.AIR import norm_features, denorm_features
from utils.draw import Artist
from user.constants import KINECT_FRAME_RATE, TARGET_FRAME_RATE
from user.constants import ALL_SUBACTION_NAMES
from user.constants import sub_action_mapping_1, sub_action_mapping_2, sub_action_mapping_3, sub_action_mapping_4
from user.constants import sub_action_mapping_5, sub_action_mapping_6
from setting import gen_sequence, TRAIN_PATH, TEST_PATH, K_MEANS_MODEL_PATH, NORM_METHOD


SEQ_LENGTH = 15


class KMeansClustering:
    def __init__(self, actions, n_clusters, seq_length, norm_method, train_paths, model_path):
        self.actions = actions
        self.n_clusters = n_clusters
        self.seq_length = seq_length
        self.norm_method = norm_method
        self.train_paths = train_paths
        self.model_path = model_path

        model_file = os.path.join(self.model_path,
                                  f"{''.join(self.actions)}_full_{self.n_clusters}_cluster.pkl")
        if not os.path.exists(model_file):
            print("K-means clustering model training...")
            self.train(actions, n_clusters)
        self.km_model = pickle.load(open(model_file, "rb"))

    def make_null_dataframe(self, input_length):
        # feature names
        # fAd == distance between human and robot at frame A
        # fAjB == position of joint B at frame A
        feature_name = [F"f{a}d" for a in range(input_length)]  # f0d ~ f14d
        feature_name.extend(
            [F"f{a}j{b+1}" for a in range(input_length) for b in range(30)])  # f0j1 ~ f0j30, f1j1 ~ f1j30, ...

        return pd.DataFrame(columns=feature_name)

    # 학습할 수 있는 DataFrame 형식으로 만들기
    def make_dataframe(self, inputs, input_length):
        df = self.make_null_dataframe(input_length)
        for i, a in enumerate(inputs):  # for each file
            temp = dict()
            for j, b in enumerate(a):  # for each sequence of 15 frames
                for k, c in enumerate(b):  # for each frame
                    if k == 0:
                        temp[F"f{j}d"] = c  # distance
                    else:
                        temp[F"f{j}j{k}"] = c
            df = df.append(temp, ignore_index=True)

        # additional features (vj0: difference between the joint positions of the first and last frames
        for i in range(input_length):
            df[F"vj{i}"] = df[F"f0j{i+1}"] - df[F"f{input_length-1}j{i+1}"]

        return df

    def preprocessing(self, input_length, file_names):
        human_data = list()
        robot_data = list()
        third_data = list()

        pbar = tqdm(total=len(file_names))
        for file in file_names:
            with np.load(file, allow_pickle=True) as data:
                human_data.append([norm_features(human, self.norm_method) for human in data['human_info']])
                robot_data.append([norm_features(robot, self.norm_method) for robot in data['robot_info']])
                third_data.append(data['third_info'])
            pbar.update(1)
        pbar.close()

        step = round(KINECT_FRAME_RATE / TARGET_FRAME_RATE)
        inputs = list()
        for idx, third in enumerate(third_data):
            if all(v == 1.0 for v in third):
                continue

            sampled_human_seq = human_data[idx][::step]
            sampled_third_seq = third_data[idx][::step]
            for human_seq, third_seq in zip(gen_sequence(sampled_human_seq, input_length),
                                            gen_sequence(sampled_third_seq, input_length)):
                inputs.append(np.concatenate((third_seq, human_seq), axis=1))

        return self.make_dataframe(inputs, input_length)

    def train(self, actions, n_clusters):
        # gather all data in train_path
        train = self.make_null_dataframe(self.seq_length)
        for action in actions:
            print(F"\nAction: {action}")
            for train_path in self.train_paths:
                files = glob.glob(os.path.join(train_path, f"*{action}*.npz"))
                train = train.append(self.preprocessing(self.seq_length, files), ignore_index=True, sort=False)
                print(f'Data loaded. Total {len(files)} files.')
        print(f'Total data size: {train.size}')

        # K-means clustering
        print('\nK-means clustering...')
        km = KMeans(n_clusters=n_clusters, random_state=2020)
        km.fit(train)

        # save model
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        with open(F"{self.model_path}/{''.join(self.actions)}_full_{self.n_clusters}_cluster.pkl", "wb") as f:
            pickle.dump(km, f)
        print('Model saved.')


def load_model(actions, n_clusters, seq_length):
    kmeans = KMeansClustering(actions=actions, n_clusters=n_clusters, seq_length=seq_length, norm_method=NORM_METHOD,
                              train_paths=[TRAIN_PATH, TEST_PATH], model_path=K_MEANS_MODEL_PATH)
    return kmeans


kmeans_1 = load_model(["A001", "A004"], 4, SEQ_LENGTH)
kmeans_2 = load_model(["A003", "A004", "A005"], 8, SEQ_LENGTH)
kmeans_3 = load_model(["A004", "A005", "A006"], 8, SEQ_LENGTH)
kmeans_4 = load_model(["A004", "A005", "A008"], 7, SEQ_LENGTH)
kmeans_5 = load_model(["A004", "A007"], 4, SEQ_LENGTH)
kmeans_6 = load_model(["A004", "A005"], 4, SEQ_LENGTH)
def load_proper_model(action):
    if action == 'A001':
        kmeans = kmeans_1
        sub_action_mapping = sub_action_mapping_1
    elif action == 'A003':
        kmeans = kmeans_2
        sub_action_mapping = sub_action_mapping_2
    elif action == 'A007':
        kmeans = kmeans_5
        sub_action_mapping = sub_action_mapping_5
    elif action == 'A004' or action == 'A005':
        kmeans = kmeans_6
        sub_action_mapping = sub_action_mapping_6
    elif action == 'A006':
        kmeans = kmeans_3
        sub_action_mapping = sub_action_mapping_3
    elif action == 'A008':
        kmeans = kmeans_4
        sub_action_mapping = sub_action_mapping_4

    return kmeans, sub_action_mapping


def test():
    # action list to test
    # actions = ["A001", "A003", "A004", "A005", "A006", "A008"]
    actions = ["A004", "A005"]

    # show all test data
    data_files = list()
    for action in actions:
        data_files.extend(glob.glob(os.path.join(TRAIN_PATH, F"*{action}*.npz")))
    data_files.sort()
    n_data = len(data_files)

    print('There are %d data.' % n_data)
    for data_idx in range(n_data):
        print('%d: %s' % (data_idx, os.path.basename(data_files[data_idx])))

    # select data name to draw
    artist = Artist(n_plot=1)
    while True:
        var = int(input("Input data number to display: "))
        data_file = data_files[var]

        with np.load(data_file, allow_pickle=True) as data:
            # action class mapping
            print(os.path.basename(data_file))
            action = os.path.basename(data_file)[4:8]
            kmeans, sub_action_mapping = load_proper_model(action)
            km_model = kmeans.km_model

            # extract inputs from data file
            human_data = [norm_features(human, NORM_METHOD) for human in data['human_info']]
            third_data = data['third_info']

            sampled_human_data = human_data[::3]
            sampled_third_data = third_data[::3]

            # draw data from start start
            for f in range(SEQ_LENGTH - 1):
                features = denorm_features(sampled_human_data[f], NORM_METHOD)
                action_info = "None"
                frame_info = f"{f+1}/{len(sampled_human_data)}"
                artist.update([features], [action_info], [frame_info], fps=10)

            # recognize sub-action
            for human_seq, third_seq in zip(gen_sequence(sampled_human_data, SEQ_LENGTH),
                                            gen_sequence(sampled_third_data, SEQ_LENGTH)):
                seq = np.concatenate((third_seq, human_seq), axis=1)
                df = kmeans.make_dataframe([seq], SEQ_LENGTH)
                sub_action = km_model.predict(df)
                action_name = ALL_SUBACTION_NAMES[sub_action_mapping[sub_action[0]]]
                print(action_name)

                f += 1
                features = denorm_features(human_seq[-1], NORM_METHOD)
                frame_info = f"{f+1}/{len(sampled_human_data)}"
                artist.update([features], [action_name], [frame_info], fps=10)


def test_all():
    # action list to test
    actions = ["A008"]

    # show all test data
    data_files = list()
    for action in actions:
        data_files.extend(glob.glob(os.path.join(TRAIN_PATH, F"*{action}*.npz")))
    for action in actions:
        data_files.extend(glob.glob(os.path.join(TEST_PATH, F"*{action}*.npz")))
    data_files.sort()
    n_data = len(data_files)

    print('There are %d data.' % n_data)

    # test each data
    for data_file in data_files:
        with np.load(data_file, allow_pickle=True) as data:
            results = list()

            # action class mapping
            # print(os.path.basename(data_file))
            action = os.path.basename(data_file)[4:8]
            kmeans, sub_action_mapping = load_proper_model(action)
            km_model = kmeans.km_model

            # extract inputs from data file
            human_data = [norm_features(human, NORM_METHOD) for human in data['human_info']]
            third_data = data['third_info']

            sampled_human_data = human_data[::3]
            sampled_third_data = third_data[::3]

            # recognize sub-action
            for human_seq, third_seq in zip(gen_sequence(sampled_human_data, SEQ_LENGTH),
                                            gen_sequence(sampled_third_data, SEQ_LENGTH)):
                seq = np.concatenate((third_seq, human_seq), axis=1)
                df = kmeans.make_dataframe([seq], SEQ_LENGTH)
                sub_action = km_model.predict(df)
                results.append(sub_action_mapping[sub_action[0]])

            # print results
            # if not all(result == 0 for result in results):
            # if not any(result == 1 for result in results):
            print(os.path.basename(data_file))
            print(results)


if "__main__" == __name__:
    test_all()
