# -*- coding: utf-8 -*-
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from user.model import Act2Act
from user.data import AIRDataSet
from user.constants import SUBACTION_NAMES
from setting import TRAIN_PATH, TEST_PATH, LSTM_MODEL_PATH

# argument parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='mode to run', choices=['train', 'verify'], required=True)
    args = parser.parse_args()

# define model parameters
lstm_input_length = 15
lstm_input_size = 25
batch_size = 64
hidden_size = 1024
output_dim = len(SUBACTION_NAMES)
learning_rate = 0.01
num_epochs = 200
save_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# create initial model
def create_model():
    model = Act2Act(device, lstm_input_size, hidden_size, output_dim)
    model.to(device)
    return model


# load trained model
def load_model(path):
    if os.path.exists(path):
        print("Load model: ", path)
        model = create_model()
        model.load_state_dict(torch.load(path))
        return model
    else:
        print("Model path is wrong: ", path)
        return


# behavior classification
def classify(model, input_batch):
    if type(input_batch) == list:
        input_batch = torch.FloatTensor(input_batch)
    input_batch = input_batch.to(device)
    scores = model(input_batch)
    predictions = torch.argmax(scores, dim=1)
    behaviors = [pred.item() for pred in predictions]
    bahevior_names = [SUBACTION_NAMES[behavior] for behavior in behaviors]
    return behaviors, bahevior_names


def load_train_data():
    print("Training data loading...")
    train_dataset = AIRDataSet(data_path=TRAIN_PATH,
                               dim_input=(lstm_input_length, lstm_input_size),
                               dim_output=(output_dim, 1))
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataset, train_data_loader


def load_test_data(data_name=None):
    print("Test data loading...")
    valid_dataset = AIRDataSet(data_path=TEST_PATH,
                               data_name=data_name,
                               dim_input=(lstm_input_length, lstm_input_size),
                               dim_output=(output_dim, 1))
    if data_name is None:
        valid_data_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return valid_dataset, valid_data_loader
    else:
        return valid_dataset


# train model
def train():
    model = create_model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # load data set
    _, train_data_loader = load_train_data()
    _, valid_data_loader = load_test_data()

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
            model_path = os.path.join(LSTM_MODEL_PATH, f"model_{epoch:04d}.pth")
            torch.save(model.state_dict(), model_path)


# show accuracy, loss, and confusion matrix of a trained model
def verify():
    # load test data
    _, valid_data_loader = load_test_data()

    # show all existing models
    model_files = glob.glob(os.path.join(LSTM_MODEL_PATH, "*.pth"))
    model_names = list()
    for model_file in model_files:
        model_name, _ = os.path.splitext(os.path.basename(model_file))
        model_names.append(model_name)
    print(f'There are {len(model_files)} models.')
    for i in range(len(model_files)):
        print(model_names[i])

    # select model and show results
    while True:
        model_num = input(f"\nInput model number to test. \nIf you want to load 'model_0010', enter '10':")
        try:
            model_num = int(model_num)
            selected_model_path = os.path.join(LSTM_MODEL_PATH, f"model_{model_num:04d}.pth")
            model = load_model(selected_model_path)
        except Exception as e:
            print(e)
            continue

        if model is not None:
            # calculate accuracy, loss, confusion matrix
            loss_function = nn.CrossEntropyLoss()
            valid_loss = 0.0
            valid_acc = 0.0
            conf_matrix = np.zeros([14, 14])
            with torch.no_grad():
                for valid_inputs, valid_outputs in valid_data_loader:
                    valid_inputs = valid_inputs.to(device)
                    valid_outputs = valid_outputs.to(device)

                    # accuracy
                    valid_scores = model(valid_inputs)
                    valid_predictions = torch.argmax(valid_scores, dim=1)
                    acc = (valid_predictions == valid_outputs).float().mean()
                    valid_acc += acc.item()

                    # loss
                    loss = loss_function(valid_scores, valid_outputs)
                    valid_loss += loss.item()

                    # confusion matrix
                    for i, j in zip(valid_outputs, valid_predictions):
                        conf_matrix[i][j] += 1

            # print average loss and accuracy
            print(f"Validation Loss: {valid_loss / len(valid_data_loader):.5f}, "
                  f"Validation Acc: {valid_acc / len(valid_data_loader):.5f}")

            # show confusion matrix
            for i in range(len(SUBACTION_NAMES)):
                n_sum = sum(conf_matrix[i])
                for j in range(len(SUBACTION_NAMES)):
                    conf_matrix[i][j] = conf_matrix[i][j] / n_sum

            labels = ["\n".join(wrap(item, 12)) for item in list(dict.values(SUBACTION_NAMES))]
            plt.figure(figsize=(13, 10))
            g = sns.heatmap(conf_matrix, annot=True, fmt='.0%', xticklabels=labels, yticklabels=labels, linewidth=1)
            g.set_xticklabels(g.get_xticklabels(), horizontalalignment='right', rotation=55)
            g.set_yticklabels(g.get_yticklabels(), horizontalalignment='right', rotation=0)
            plt.xlabel('Predicted label', fontsize=20)
            plt.ylabel('True label', fontsize=20)
            # plt.title('Confusion Matrix\n', fontsize=25)
            plt.tight_layout()
            plt.show()


def main():
    if args.mode == "train":
        train()
    if args.mode == "verify":
        verify()


if __name__ == '__main__':
    main()
