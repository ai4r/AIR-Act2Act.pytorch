# -*- coding: utf-8 -*-
# AIR-Act2Act 데이터셋을 불러와서 data augmentation 하기
# data augmentation 1: skeleton 정보에 random noise 추가
# data augmentation 2: action 간 부드럽게 연결된 데이터 생성
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


# 수정해야할 코드
class AIRDataSet:
    def __init__(self):
        pass

    def add_random_noise(self):
        pass

    def connect_sequence(self):
        pass


# 아래는 샘플 코드
class SampleDataset:
    def __init__(self):
        self.training_data = [
            ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
            ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
        ]
        self.word_to_ix = {}
        for sent, tags in self.training_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
        print(self.word_to_ix)
        self.tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

        # These will usually be more like 32 or 64 dimensional.
        # We will keep them small, so we can see how the weights change as we train.
        self.EMBEDDING_DIM = 6
        self.HIDDEN_DIM = 6

    def prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        tensor = torch.LongTensor(idxs)
        return autograd.Variable(tensor)
