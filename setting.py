import os


# function to extract sequence from data
def gen_sequence(data, length):
    for start_idx in range(len(data) - length + 1):
        yield list(data[start_idx:start_idx + length])


# skeleton features
INPUT_DATA_TYPE = '3D'  # {'2D', '3D'}
NORM_METHOD = 'vector'  # {'vector', 'torso'}
B_HANDS = False  # {True, False}

# actions to train or test
# ACTIONS = ["A%03d" % a for a in range(1, 11)]
ACTIONS = ['A001', 'A003', 'A004', 'A005', 'A006', 'A008']
# ACTIONS = ['A001', 'A003', 'A004', 'A005', 'A006', 'A007', 'A008']
# ACTIONS = ['A004', 'A005', 'A006', 'A007']
# ACTIONS = ['A001', 'A004', 'A005', 'A007', 'A008']

#  paths
PROB_TRAIN = 0.9
DEVIDE = 'scene'  # {'scene', 'subject'}
DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r'data files',
                         str(PROB_TRAIN), DEVIDE)
TRAIN_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), DATA_PATH, 'train data')
TEST_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), DATA_PATH, 'valid data')
LSTM_MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r'user\models\lstm',
                               ''.join(ACTIONS), NORM_METHOD, INPUT_DATA_TYPE, str(B_HANDS))
K_MEANS_MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r'user\models\k-means')

# camera setting
W_VIDEO = 640
H_VIDEO = 480
