# -*- coding: utf-8 -*-
import os
import glob
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from model import Act2Act
from data import AIRDataSet, ACTIONS, norm_method
from data import KINECT_FRAME_RATE, TARGET_FRAME_RATE, gen_sequence
from utils.AIR import norm_features, denorm_features
from utils.draw import draw


# define model parameters
lstm_input_length = 30
lstm_input_size = 25
batch_size = 64
hidden_size = 1024
output_dim = len(ACTIONS)
learning_rate = 0.01
num_epochs = 200
save_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define LSTM model
model = Act2Act(device, lstm_input_size, hidden_size, output_dim)
model.to(device)

# model and data path
MODEL_PATH = './models/recog_action'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
TEST_PATH = './data files/valid data/'


def train():
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # load data set
    train_dataset = AIRDataSet(data_path='./data files/train data',
                               dim_input=(lstm_input_length, lstm_input_size),
                               dim_output=(output_dim, 1))
    valid_dataset = AIRDataSet(data_path='./data files/valid data',
                               dim_input=(lstm_input_length, lstm_input_size),
                               dim_output=(output_dim, 1))
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_data_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # training
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        train_acc = 0.0
        for train_inputs, train_outputs in train_data_loader:
            model.zero_grad()

            train_inputs = train_inputs.to(device)
            train_outputs = train_outputs.to(device)

            train_scores = model(train_inputs)
            train_predictions = torch.argmax(train_scores, dim=1)
            acc = (train_predictions == train_outputs).float().mean()
            train_acc += acc.item()

            loss = loss_function(train_scores, train_outputs)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        # validation
        if epoch % save_epochs == 0:
            valid_loss = 0.0
            valid_acc = 0.0
            with torch.no_grad():
                for valid_inputs, valid_outputs in valid_data_loader:
                    valid_inputs = valid_inputs.to(device)
                    valid_outputs = valid_outputs.to(device)

                    valid_scores = model(valid_inputs)
                    valid_predictions = torch.argmax(valid_scores, dim=1)
                    acc = (valid_predictions == valid_outputs).float().mean()
                    valid_acc += acc.item()

                    loss = loss_function(valid_scores, valid_outputs)
                    valid_loss += loss.item()

            # Print average training/validation loss and accuracy
            print(f"Epoch {epoch}")
            print(f"Training Loss: {train_loss / len(train_data_loader):.5f}, "
                  f"Training Acc: {train_acc / len(train_data_loader):.5f}")
            print(f"Validation Loss: {valid_loss / len(valid_data_loader):.5f}, "
                  f"Validation Acc: {valid_acc / len(valid_data_loader):.5f}")
            model_path = os.path.join(MODEL_PATH, f"model_{epoch:04d}.pth")
            torch.save(model.state_dict(), model_path)


def test():
    # train model if not exists
    model_files = glob.glob(os.path.join(MODEL_PATH, "*.pth"))
    if len(model_files) == 0:
        train()

    # show all existing models
    model_names = list()
    for model_file in model_files:
        model_name, _ = os.path.splitext(os.path.basename(model_file))
        model_names.append(model_name)
    print(f'There are {len(model_files)} models ({model_names[0]} ~ {model_names[-1]}).')

    # select model to test
    while True:
        model_num = input(f"Input model number to test ({model_names[0][6:]} ~ {model_names[-1][6:]}): ")
        model_num = int(model_num)
        selected_model = os.path.join(MODEL_PATH, f"model_{model_num:04d}.pth")
        if os.path.exists(selected_model):
            print("Load model: ", selected_model)
            model.load_state_dict(torch.load(selected_model))
            break
        print("Model number is wrong.")

    # load test data
    data_files = list()
    for action in ACTIONS:
        data_files.extend(glob.glob(os.path.join(TEST_PATH, f"*{action}*.npz")))
    random.shuffle(data_files)
    print(f'There are {len(data_files)} data.')
    for idx in range(min(len(data_files), 20)):
        data_file = os.path.basename(data_files[idx])
        data_name, _ = os.path.splitext(data_file)
        print(f'{idx}: {data_name}')

    # select data to test
    while True:
        var = int(input("Input data number to display: "))
        test_file = data_files[var]
        print(os.path.normpath(test_file))

        with np.load(test_file, allow_pickle=True) as data:
            human_data = [norm_features(human, norm_method) for human in data['human_info']]
            third_data = data['third_info']
            step = round(KINECT_FRAME_RATE / TARGET_FRAME_RATE)
            sampled_human_data = human_data[::step]
            sampled_third_data = third_data[::step]

            # ground truth
            action_name = [action for action in ACTIONS if action in test_file][0]
            cur_action = ACTIONS.index(action_name)

            # prediction results
            outputs = list()
            predictions = list()
            for human_seq, third_seq in zip(gen_sequence(sampled_human_data, lstm_input_length),
                                            gen_sequence(sampled_third_data, lstm_input_length)):
                inputs = np.concatenate((third_seq, human_seq), axis=1)
                input_batch = torch.FloatTensor([inputs]).to(device)
                scores = model(input_batch)
                prediction = torch.argmax(scores, dim=1)
                predictions.append(prediction.item())
                outputs.append(cur_action)

            print("true: ", outputs)
            print("pred: ", predictions)

            # draw results
            features = list()
            for f in range(len(sampled_human_data)):
                cur_features = sampled_human_data[f]
                cur_features = denorm_features(cur_features, norm_method)
                features.append(cur_features)

            predictions = ["None"] * (lstm_input_length - 1) + predictions

            draw([features], [predictions], save_path=None, b_show=True)


if __name__ == '__main__':
    test()
