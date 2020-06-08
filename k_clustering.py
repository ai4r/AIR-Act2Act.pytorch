import os
import glob
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans

from utils.AIR import norm_features, denorm_features
from utils.draw import draw
from constants import KINECT_FRAME_RATE, TARGET_FRAME_RATE, gen_sequence
from constants import SUBACTION_NAMES, sub_action_mapping_1, sub_action_mapping_2, sub_action_mapping_3


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
            self.train(actions, n_clusters)
        self.km_model = pickle.load(open(model_file, "rb"))

    def make_null_dataframe(self, input_length):
        # feature 이름 생성
        # fAd == 해당 시퀀스의 frame A번째의 사람과 로봇사이 distance
        # fAjB == 해당 시퀀스의 frame A번째의 B번째 joint
        feature_name = [F"f{a}d" for a in range(input_length)]  # f0d ~ f19d
        feature_name.extend(
            [F"f{a}j{b+1}" for a in range(input_length) for b in range(24)])  # f0j1 ~ f0j24, f1j1 ~ f1j24, ...

        return pd.DataFrame(columns=feature_name)

    # 학습할 수 있는 DataFrame 형식으로 만들기
    def make_dataframe(self, inputs, input_length):
        df = self.make_null_dataframe(input_length)
        for i, a in enumerate(inputs):  # 파일들에 대해
            temp = dict()
            for j, b in enumerate(a):  # 20frame짜리 시퀀스에 대해
                for k, c in enumerate(b):  # 프레임에 대해
                    if k == 0:
                        temp[F"f{j}d"] = c  # 프레임의 거리정보
                    else:
                        temp[F"f{j}j{k}"] = c
            df = df.append(temp, ignore_index=True)

        # 새로운 분산 feature 추가.  vj0 == 해당 시퀀스의 첫번째 frame과 마지막 frame에서 0번째 joint의 차이
        for i in range(20):
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
        # 모든 train_path, action에 대해 테이터를 모으기
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

        # 학습한 model save
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        with open(F"{self.model_path}/{''.join(self.actions)}_full_{self.n_clusters}_cluster.pkl", "wb") as f:
            pickle.dump(km, f)
        print('Model saved.')


def test():
    # 보고싶은 액션 입력
    actions = ["A001", "A004"]
    n_clusters = 4

    # actions = ["A004", "A005", "A006"]
    # n_clusters = 9

    # actions = ["A004", "A005", "A008"]
    # n_clusters = 7

    # parameters
    seq_length = 15
    norm_method = 'vector'
    train_path = './data files/train data'
    test_path = './data files/valid data'
    model_path = './models/k-means'

    kmeans = KMeansClustering(actions=actions, n_clusters=n_clusters, seq_length=seq_length, norm_method=norm_method,
                              train_paths=[train_path, test_path], model_path=model_path)
    km_model = kmeans.km_model

    # show all test data
    data_files = list()
    for action in actions:
        data_files.extend(glob.glob(os.path.join(test_path, F"*{action}*.npz")))
    data_files.sort()
    n_data = len(data_files)

    print('There are %d data.' % n_data)
    for data_idx in range(n_data):
        print('%d: %s' % (data_idx, os.path.basename(data_files[data_idx])))

    # select data name to draw
    while True:
        var = int(input("Input data number to display: "))
    # for var in range(n_data):
        data_file = data_files[var]

        with np.load(data_file, allow_pickle=True) as data:
            print(os.path.basename(data_file))
            action = os.path.basename(data_file)[4:8]

            if action == 'A001' or action == 'A004':
                sub_action_mapping = sub_action_mapping_1
            elif action == 'A005' or action == 'A006':
                sub_action_mapping = sub_action_mapping_2
            elif action == 'A008':
                sub_action_mapping = sub_action_mapping_3

            human_data = [norm_features(human, norm_method) for human in data['human_info']]
            third_data = data['third_info']

            sampled_human_data = human_data[::3]
            sampled_third_data = third_data[::3]

            # recognize sub-action
            predictions = list()
            for human_seq, third_seq in zip(gen_sequence(sampled_human_data, seq_length),
                                            gen_sequence(sampled_third_data, seq_length)):
                seq = np.concatenate((third_seq, human_seq), axis=1)
                df = kmeans.make_dataframe([seq], seq_length)
                sub_action = km_model.predict(df)
                predictions.append(sub_action[0])
            print(predictions)

            # draw results
            features = list()
            for f in range(len(sampled_human_data)):
                cur_features = sampled_human_data[f]
                cur_features = denorm_features(cur_features, norm_method)
                features.append(cur_features)
            names = [SUBACTION_NAMES[sub_action_mapping[pred]] for pred in predictions]
            predictions = ["None"] * (seq_length - 1) + names
            draw([features], [predictions], save_path=None, b_show=True)


if "__main__" == __name__:
    test()
