# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn


class Act2Act(nn.Module):
    def __init__(self, device, input_size, hidden_size, output_size):
        super(Act2Act, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        hidden, cell = self.init_hidden(inputs.size(0))
        output, (hidden, cell) = self.lstm(inputs, (hidden, cell))

        hidden = hidden[-1:]
        hidden = torch.cat([h for h in hidden], 1)
        output = self.fc(hidden)
        # output = self.softmax(output)

        return output

    def init_hidden(self, batch_size):
        hidden = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).to(self.device))
        cell = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).to(self.device))
        return hidden, cell
