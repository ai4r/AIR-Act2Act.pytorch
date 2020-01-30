import os
import glob
import numpy as np
from tqdm import tqdm

from utils.AIR import read_joint


joint_path = './joint files/'
data_path = './data files/'

human_files = glob.glob(os.path.normpath(os.path.join(joint_path, "C001*.joint")))
human_files.sort()
n_data = len(human_files)

pbar = tqdm(total=n_data)
for human_file in human_files:
    robot_file = human_file.replace('C001', 'C002')
    third_file = human_file.replace('C001', 'C003')

    human_info = read_joint(human_file)
    robot_info = read_joint(robot_file)
    third_info = read_joint(third_file)

    data_name = human_file.replace("\\", "/").split("/")[-1].split('.')[0]
    data_file = data_path + f"{data_name[4:]}.npz"
    np.savez(data_file, human_info=human_info, robot_info=robot_info, third_info=third_info)

    pbar.update(1)

pbar.close()
