# -*- coding: utf-8 -*-
# Act2Act 딥러닝 모델 구조를 정의
# 원래는 {input: 사람 skeleton sequence, output: 로봇 joint angle sequence} 이지만,
# 단순한 문제를 먼저 풀기 위해서
# {input: 사람 skeleton sequence, output: 사람 action class} 로 가정
import torch
import torch.autograd as autograd
import torch.nn as nn

import random


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, device, n_layers=1, dropout=.2):
        super(Encoder, self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.dropout(inputs)

        hidden, cell = self.init_hidden(inputs.size(0))
        output, (hidden, cell) = self.lstm(inputs, (hidden, cell))

        return hidden, cell

    def init_hidden(self, batch_size):
        hidden = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).to(self.device))
        cell = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).to(self.device))

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1, dropout=.2):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(output_size, hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        input = self.dropout(input)

        output, (hidden, cell) = self.lstm(input, (hidden, cell))

        prediction = self.fc(output)
        return prediction, hidden, cell


# seq2seq model for generation of robot action
class Act2Act(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Act2Act, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, encoder_inputs, decoder_input, decoder_outputs, teacher_forcing_ratio=0.5):
        batch_size = decoder_outputs.shape[0]
        output_length = decoder_outputs.shape[1]
        output_size = self.decoder.output_size
        outputs = torch.zeros(batch_size, output_length, output_size).to(self.device)

        hidden, cell = self.encoder(encoder_inputs)

        for t in range(output_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                decoder_input = decoder_outputs[:, t, :].unsqueeze(1)
            else:
                decoder_input = output

        return outputs
