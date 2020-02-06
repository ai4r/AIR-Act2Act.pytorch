# -*- coding: utf-8 -*-
# 딥러닝 모델 학습
# 아래는 샘플 코드 (수정 필요)
from model import Act2Act
from data import AIRDataSet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


# 수정해야할 코드
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Act2Act(25, 1024, 2)
    model.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    batch_size = 64
    dataset = AIRDataSet(data_path='./data files', dim_input=(30, 25), dim_output=(2, 1))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1000):
        for inputs, outputs in data_loader:
            model.zero_grad()

            inputs = inputs.to(device)
            outputs = outputs.to(device)

            scores = model(inputs)
            loss = loss_function(scores, outputs)
            loss.backward()
            optimizer.step()
            print(loss)


if __name__ == "__main__":
    main()