# -*- coding: utf-8 -*-
from model import Act2Act
from data import AIRDataSet, ACTIONS

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


def main():
    lstm_input_length = 30
    lstm_input_size = 25
    batch_size = 64
    hidden_size = 1024
    output_dim = len(ACTIONS)
    learning_rate = 0.01
    num_epochs = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define LSTM model
    model = Act2Act(lstm_input_size, hidden_size, output_dim)
    model.cuda()
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
        if epoch % 10 == 0:
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
            model_path = f'./models/model_{epoch:04d}.pth'
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
