import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.spatial.distance import cdist

from utils.AIR import norm_to_torso
from data import gen_sequence


KINECT_FRAME_RATE = 30  # frame rate of kinect camera
TARGET_FRAME_RATE = 10  # frame rate of extracted data
# ACTIONS = ["A%03d" % a for a in range(1, 11)]
ACTIONS = [f'A00{x}' for x in range(1, 10)]
ACTIONS.append("A010")
train_path = './data files/train data'
test_path = './data files/valid data'
model_path = './models/k-means'

# 알맞은 k값 넣기
N_CLUSTERS = [3, 3, 3, 5, 3, 4, 4, 4, 5, 3]
SEQ_LENGTH = 15


# elbow 값과 실루엣 값을 얻는 함수
def get_kmeans_insight(df):
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.rcParams['lines.linewidth'] = 10
    plt.rcParams['lines.color'] = 'r'
    plt.rcParams['axes.grid'] = True
    distortions = []
    ks = range(1, 10)
    for k in ks:
        km_model = KMeans(n_clusters=k, random_state=2020).fit(df)
        distortions.append(sum(np.min(cdist(df, km_model.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
        cluster_labels = km_model.labels_
        if k != 1:
            # silhouette score
            silhouette_avg = metrics.silhouette_score(df, cluster_labels)
            print("For n_clusters={0}, the silhouette score is {1}".format(k, silhouette_avg))
    # Plot the elbow
    plt.plot(ks, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Result')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


# k-means clustering 결과값 그래프로 확인하기
def draw_plots(km_values):
    plt.rcParams["figure.figsize"] = (500, 10)
    plt.rcParams['lines.linewidth'] = 10
    plt.rcParams['lines.color'] = 'r'
    plt.rcParams['axes.grid'] = True
    for i in range(0, len(km_values), 500):
        plt.plot(km_values[i:i+500])
        plt.show()


# 학습할 수 있는 DataFrame 형식으로 만들기
def make_dataframe(inputs, input_length):
    # feature 이름 생성
    # fAd == 해당 시퀀스의 frame A번째의 사람과 로봇사이 distance
    # fAjB == 해당 시퀀스의 frame A번째의 B번째 joint
    feature_name = [F"f{a}d" for a in range(input_length)]  # f0d ~ f19d
    feature_name.extend(
        [F"f{a}j{b+1}" for a in range(input_length) for b in range(24)])  # f0j1 ~ f0j24, f1j1 ~ f1j24, ...

    df = pd.DataFrame(columns=feature_name)
    pbar = tqdm(total=len(inputs))
    for i, a in enumerate(inputs):  # 파일들에 대해
        temp = dict()
        for j, b in enumerate(a):  # 20frame짜리 시퀀스에 대해
            for k, c in enumerate(b):  # 프레임에 대해
                if k == 0:
                    temp[F"f{j}d"] = c  # 프레임의 거리정보
                else:
                    temp[F"f{j}j{k}"] = c
        df = df.append(temp, ignore_index=True)
        pbar.update(1)
    pbar.close()

    # 새로운 분산 feature 추가.  vj0 == 해당 시퀀스의 첫번째 frame과 마지막 frame에서 0번째 joint의 차이
    for i in range(20):
        df[F"vj{i}"] = df[F"f0j{i+1}"] - df[F"f{input_length-1}j{i+1}"]

    return df


def preprocessing(input_length, file_names):
    human_data = list()
    robot_data = list()
    third_data = list()

    pbar = tqdm(total=len(file_names))
    for file in file_names:
        with np.load(file, allow_pickle=True) as data:
            human_data.append([norm_to_torso(human) for human in data['human_info']])
            robot_data.append([norm_to_torso(robot) for robot in data['robot_info']])
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

    return make_dataframe(inputs, input_length)


# DataFrame에서 null값이 있는 column들의 리스트를 반환합니다.
def check_null(df):
    col_list = []
    for col in df.columns:
        if df[col].isnull().values.any():
            col_list.append(col)
    return col_list


def main():
    for i, act in enumerate(ACTIONS):
        print(F"\n\nAction: {act}")
        train_files = glob.glob(os.path.join(train_path, f"*{act}*.npz"))
        print(f'Train data loading. Total {len(train_files)} files.')
        test_files = glob.glob(os.path.join(test_path, f"*{act}*.npz"))
        print(f'Test data loading. Total {len(test_files)} files.')

        print('Preprocessing...')
        train = preprocessing(SEQ_LENGTH, train_files)
        test = preprocessing(SEQ_LENGTH, test_files)

        # 전체 데이터를 보기 위해서 train, test set 합침
        train = pd.concat([train, test], axis=0)

        # null값 있는지 체크
        #     if( len(check_null(train)) == 0 and check_null(test) == 0 ):
        #         print("null value existed")
        #         break

        # 적절한 k값 구하기
        #     get_kmeans_insight(train)

        print('K-means clustering...')
        km = KMeans(n_clusters=N_CLUSTERS[i], random_state=2020)
        km.fit(train)

        # 결과값 그래프 그리기
        #     print(F"{act}의 train")
        #     draw_plots(km.labels_)

        #     print(F"{act}의 test")
        #     draw_plots(km.predict(test))

        # 학습한 model save
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        with open(F"{model_path}/{act}_full_{N_CLUSTERS[i]}_cluster.pkl", "wb") as f:
            pickle.dump(km, f)
        print('Model saved.')


if "__main__" == __name__:
    main()
