import os


# function to extract sequence from data
def gen_sequence(data, length):
    for start_idx in range(len(data) - length + 1):
        yield list(data[start_idx:start_idx + length])


# skeleton features
INPUT_DATA_TYPE = '2D'  # {'2D', '3D'}
NORM_METHOD = 'vector'  # {'vector', 'torso'}
B_HANDS = False  # {True, False}

# actions to train or test
# ACTIONS = ["A%03d" % a for a in range(1, 11)]
# ACTIONS = ['A001', 'A003', 'A004', 'A005', 'A006', 'A008']
# ACTIONS = ['A001', 'A003', 'A004', 'A005', 'A006', 'A007', 'A008']
# ACTIONS = ['A004', 'A005', 'A006', 'A007']
ACTIONS = ['A001', 'A004', 'A005', 'A007', 'A008']

#  paths
PROB_TRAIN = 0.9
DEVIDE = 'scene'  # {'scene', 'subject'}
PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PATH, 'data files', str(PROB_TRAIN), DEVIDE)
JOINT_PATH = os.path.join(PATH, 'joint files')
TRAIN_PATH = os.path.join(DATA_PATH, 'train data')
TEST_PATH = os.path.join(DATA_PATH, 'valid data')

MODEL_PATH = os.path.join(PATH, 'user', 'models')
LSTM_MODEL_PATH = os.path.join(MODEL_PATH, 'lstm', ''.join(ACTIONS), INPUT_DATA_TYPE, NORM_METHOD, str(B_HANDS))
K_MEANS_MODEL_PATH = os.path.join(MODEL_PATH, 'k-means')

# camera setting
W_VIDEO = 640
H_VIDEO = 480
