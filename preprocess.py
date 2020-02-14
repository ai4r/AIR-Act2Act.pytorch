import os
import glob
import random
import shutil
import numpy as np
from tqdm import tqdm

from utils.AIR import read_joint, vectorize, move_camera_to_front

JOINT_PATH = './joint files/'
DATA_PATH = './data files/'
TRAIN_PATH = './data files/train data/'
VALID_PATH = './data files/valid data/'
MAX_DISTANCE = 5.  # maximum distance between camera and human
prob_train = 0.9


def gen_datafiles():
    human_files = glob.glob(os.path.normpath(os.path.join(JOINT_PATH, "C001*.joint")))
    human_files.sort()
    n_data = len(human_files)

    pbar = tqdm(total=n_data)
    for human_file in human_files:
        robot_file = human_file.replace('C001', 'C002')
        third_file = human_file.replace('C001', 'C003')

        if not os.path.exists(robot_file) or not os.path.exists(third_file):
            continue

        human_info = read_joint(human_file)
        robot_info = read_joint(robot_file)
        third_info = read_joint(third_file)

        extracted_human_info = list()
        extracted_robot_info = list()
        extracted_third_info = list()

        # extract distance features first
        n_frames = min(len(human_info), len(robot_info), len(third_info))
        max_value = max(max(len(human_info) - n_frames, len(robot_info) - n_frames), len(third_info) - n_frames)
        if max_value > 30:
            wer = 234
        for f in range(n_frames):
            n_body = sum(1 for b in third_info[f] if b is not None)
            if n_body != 2:
                if 'A001' not in human_file and 'A010' not in human_file:
                    raise Exception(f'third camera information is wrong. ({third_file})')

                extracted_third_info.append([MAX_DISTANCE / MAX_DISTANCE])
                continue

            robot_pos1 = vectorize(third_info[f][0]["joints"][0])
            human_pos1 = vectorize(third_info[f][1]["joints"][0])
            dist_third = MAX_DISTANCE if all(v == 0 for v in human_pos1) else np.linalg.norm(human_pos1 - robot_pos1)

            dist_human = MAX_DISTANCE
            if human_info[f][1] is not None:
                human_pos2 = vectorize(human_info[f][1]["joints"][0])
                robot_pos2 = np.array([0., 0., 0.])
                dist_human = MAX_DISTANCE if all(v == 0 for v in human_pos2) else np.linalg.norm(human_pos2 - robot_pos2)

            dist = min(dist_third, dist_human)
            extracted_third_info.append([dist / MAX_DISTANCE])

        # move camera position in front of person
        move_camera_to_front(human_info, body_id=1)
        move_camera_to_front(robot_info, body_id=0)

        for f in range(n_frames):
            extracted_human_info.append(human_info[f][1]["joints"])
            extracted_robot_info.append(robot_info[f][0]["joints"])

        data_name = human_file.replace("\\", "/").split("/")[-1].split('.')[0]
        data_file = DATA_PATH + f"{data_name[4:]}.npz"
        np.savez(data_file,
                 human_info=extracted_human_info,
                 robot_info=extracted_robot_info,
                 third_info=extracted_third_info)

        pbar.update(1)

    pbar.close()


def split_train_valid():
    # split data set into train data and valid data
    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    if not os.path.exists(VALID_PATH):
        os.makedirs(VALID_PATH)

    reset_train = glob.glob(os.path.join(TRAIN_PATH, "*.npz"))
    for file in reset_train:
        shutil.move(file, os.path.join(DATA_PATH, os.path.basename(file)))
    reset_valid = glob.glob(os.path.join(VALID_PATH, "*.npz"))
    for file in reset_valid:
        shutil.move(file, os.path.join(DATA_PATH, os.path.basename(file)))

    files = glob.glob(os.path.join("./data files", "*.npz"))
    data_names = list()

    for file in files:
        file_name = os.path.basename(file)
        data_name = file_name[:8]
        if data_name not in data_names:
            data_names.append(data_name)

    random.shuffle(data_names)

    train = data_names[:int(len(data_names)*prob_train)]

    for file in files:
        file_name = os.path.basename(file)
        if file_name[:8] in train:
            shutil.move(file, os.path.join(TRAIN_PATH, file_name))
        else:
            shutil.move(file, os.path.join(VALID_PATH, file_name))


if __name__ == "__main__":
    split_train_valid()
