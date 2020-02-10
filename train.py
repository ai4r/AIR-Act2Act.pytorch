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
    num_epochs = 3000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define LSTM model
    model = Act2Act(lstm_input_size, hidden_size, output_dim)
    model.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # load data set
    dataset = AIRDataSet(data_path='./data files',
                         dim_input=(lstm_input_length, lstm_input_size),
                         dim_output=(output_dim, 1))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # training
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_acc = 0.0
        for inputs, outputs in data_loader:
            model.zero_grad()

            inputs = inputs.to(device)
            outputs = outputs.to(device)

            scores = model(inputs)
            predictions = torch.argmax(scores, dim=1)
            acc = (predictions == outputs).float().mean()
            total_acc += acc.item()

            loss = loss_function(scores, outputs)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        # average loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader):.5f}, Acc: {total_acc / len(data_loader):.5f}")
            model_path = f'./models/model_{epoch:04d}.pth'
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
