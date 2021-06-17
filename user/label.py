import os
import glob
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.AIR import norm_features
from utils.kcluster import load_proper_model, SEQ_LENGTH
from user.constants import ALL_SUBACTION_NAMES, KINECT_FRAME_RATE, TARGET_FRAME_RATE
from setting import gen_sequence, TRAIN_PATH, TEST_PATH, NORM_METHOD

# action list to test
# actions = ["A001", "A004", "A005", "A007", "A008"]
# actions = ["A001", "A003", "A004", "A005", "A006", "A008"]
# actions = ["A005"]
# actions = ['A001', 'A004', 'A005', 'A007', 'A008']
actions = ["A008"]

# get all data files
data_files = list()
for action in actions:
    data_files.extend(glob.glob(os.path.join( TEST_PATH, F"*{action}*.npz")))
    data_files.extend(glob.glob(os.path.join(TRAIN_PATH, F"*{action}*.npz")))
data_files.sort()

# label action classes
pbar = tqdm(total=len(data_files))
for data_file in data_files:
    with np.load(data_file, allow_pickle=True) as data:
        # action class mapping
        action = os.path.basename(data_file)[4:8]
        kmeans, sub_action_mapping = load_proper_model(action)
        km_model = kmeans.km_model

        # extract inputs from data file
        human_data = [norm_features(human, method=NORM_METHOD, type='3D', b_hand=True) for human in data['human_info']]
        third_data = data['third_info']

        step = round(KINECT_FRAME_RATE / TARGET_FRAME_RATE)
        sampled_human_data = human_data[::step]
        sampled_third_data = third_data[::step]

        # label "None"
        sampled_labels = list()
        for f in range(SEQ_LENGTH - 1):
            sampled_labels.append("None")

        # label recognized action class by k-means clustering
        for human_seq, third_seq in zip(gen_sequence(sampled_human_data, SEQ_LENGTH),
                                        gen_sequence(sampled_third_data, SEQ_LENGTH)):
            seq = np.concatenate((third_seq, human_seq), axis=1)
            df = kmeans.make_dataframe([seq], SEQ_LENGTH)
            sub_action = km_model.predict(df)
            action_name = ALL_SUBACTION_NAMES[sub_action_mapping[sub_action[0]]]
            sampled_labels.append(action_name)

        # add to data
        labels = list()
        for f in range(len(human_data)):
            labels.append(sampled_labels[int(f/step)])

        np.savez(data_file,
                 human_info=data['human_info'],
                 robot_info=data['robot_info'],
                 third_info=data['third_info'],
                 human_action=labels)

    pbar.update(1)
pbar.close()
