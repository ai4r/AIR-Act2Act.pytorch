import os


# function to extract sequence from data
def gen_sequence(data, length):
    for start_idx in range(len(data) - length + 1):
        yield list(data[start_idx:start_idx + length])

# skeleton feature normalization
NORM_METHOD = 'vector'

# actions to train or test
# ACTIONS = ["A%03d" % a for a in range(1, 11)]
ACTIONS = ['A001', 'A003', 'A004', 'A005', 'A006', 'A008']

#  paths
TRAIN_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r'data files\train data')
TEST_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r'data files\valid data')
LSTM_MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r'user\models\lstm', NORM_METHOD)
K_MEANS_MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), r'user\models\k-means')
