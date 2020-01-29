# -*- coding: utf-8 -*-
# Act2Act 딥러닝 모델 구조를 정의
# 원래는 {input: 사람 skeleton sequence, output: 로봇 joint angle sequence} 이지만,
# 단순한 문제를 먼저 풀기 위해서
# {input: 사람 skeleton sequence, output: 사람 action class} 로 가정
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


# 수정해야할 코드
class Act2Act(nn.Module):
    def __init__(self):
        super(Act2Act, self).__init__()
        pass

    def forward(self):
        pass


# 아래는 샘플 코드
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
