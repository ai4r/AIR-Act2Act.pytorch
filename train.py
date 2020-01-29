# -*- coding: utf-8 -*-
# 딥러닝 모델 학습
# 아래는 샘플 코드 (수정 필요)
from model import Act2Act, LSTMTagger
from data import AIRDataSet, SampleDataset

import torch.nn as nn
import torch.optim as optim


# 수정해야할 코드
def main():
    pass


# 아래는 샘플 코드
def sample():
    dataset = SampleDataset()
    model = LSTMTagger(dataset.EMBEDDING_DIM, dataset.HIDDEN_DIM, len(dataset.word_to_ix), len(dataset.tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    inputs = dataset.prepare_sequence(dataset.training_data[0][0], dataset.word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in dataset.training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            sentence_in = dataset.prepare_sequence(sentence, dataset.word_to_ix)
            targets = dataset.prepare_sequence(tags, dataset.tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    inputs = dataset.prepare_sequence(dataset.training_data[0][0], dataset.word_to_ix)
    tag_scores = model(inputs)
    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    #  for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)


if __name__ == "__main__":
    sample()
    # main()
