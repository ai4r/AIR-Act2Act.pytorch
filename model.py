# -*- coding: utf-8 -*-
# Act2Act 딥러닝 모델 구조를 정의
# 원래는 {input: 사람 skeleton sequence, output: 로봇 joint angle sequence} 이지만,
# 단순한 문제를 먼저 풀기 위해서
# {input: 사람 skeleton sequence, output: 사람 action class} 로 가정
import torch
import torch.autograd as autograd
import torch.nn as nn
import random


# seq2seq model for generation of robot action
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.device = device

    def forward(self, inputs):
        hidden, cell = self.init_hidden(inputs.size(0))

        output, (hidden, cell) = self.lstm(inputs, (hidden, cell))

        return hidden, cell

    def init_hidden(self, batch_size):
        hidden = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).to(self.device))
        cell = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).to(self.device))

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_, hidden, cell):
        output, (hidden, cell) = self.lstm(input_, (hidden, cell))
        prediction = self.fc(output)

        return prediction, hidden, cell


class Act2Act(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Act2Act, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, encoder_inputs, decoder_input, decoder_outputs, teacher_forcing_ratio=0):
        output_length = decoder_outputs.size(1)
        outputs = torch.zeros(decoder_outputs.size(0), decoder_outputs.size(1), decoder_outputs.size(2)).to(self.device)

        hidden, cell = self.encoder(encoder_inputs)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for t in range(1, output_length):
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                outputs[:, t-1, :] = output.squeeze(1)
                decoder_input = decoder_outputs[:, t-1, :].unsqueeze(1)
        else:
            for t in range(1, output_length):
                output, hidden, cell = self.decoder(decoder_input, hidden, cell)
                outputs[:, t-1, :] = output.squeeze(1)
                decoder_input = output

        return outputs


# LSTM model for classification of human action
# class Act2Act(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Act2Act, self).__init__()
#
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, inputs):
#         hidden, cell = self.init_hidden(inputs.size(0))
#         output, (hidden, cell) = self.lstm(inputs, (hidden, cell))
#
#         hidden = hidden[-1:]
#         hidden = torch.cat([h for h in hidden], 1)
#         output = self.fc(hidden)
#
#         return output
#
#     def init_hidden(self, batch_size):
#         hidden = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
#         cell = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
#         return hidden, cell
