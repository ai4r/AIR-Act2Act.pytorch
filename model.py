# -*- coding: utf-8 -*-
# Act2Act 딥러닝 모델 구조를 정의
# 원래는 {input: 사람 skeleton sequence, output: 로봇 joint angle sequence} 이지만,
# 단순한 문제를 먼저 풀기 위해서
# {input: 사람 skeleton sequence, output: 사람 action class} 로 가정
import torch
import torch.autograd as autograd
import torch.nn as nn

import random


class Act2Act(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Act2Act, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, encoder_inputs, decoder_inputs, decoder_outputs, teacher_forcing_ratio=0.5):
        batch_size = decoder_outputs.shape[0]
        trg_len = decoder_outputs.shape[1]
        trg_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(encoder_inputs)

        # first input to the decoder is the <sos> tokens
        inp = decoder_inputs

        for t in range(trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            out, hidden, cell = self.decoder(inp, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = out.squeeze(1)

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            inp = decoder_outputs[:, t, :].unsqueeze(1) if teacher_force else out

        return outputs


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=.2):
        super(Encoder, self).__init__()

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
        hidden = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        cell = autograd.Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

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
