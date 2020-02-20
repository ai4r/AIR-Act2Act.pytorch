# -*- coding: utf-8 -*-
from model import Act2Act, Encoder, Decoder
from data import AIRDataSet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


def main():
    lstm_input_length = 20
    lstm_input_size = 25
    lstm_output_length = 5
    lstm_output_size = 24
    batch_size = 64
    hidden_size = 1024
    learning_rate = 0.001
    num_epochs = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data set
    train_dataset = AIRDataSet(data_path='./data files/train data',
                               dim_input=(lstm_input_length, lstm_input_size),
                               dim_output=(lstm_output_length, lstm_output_size))
    valid_dataset = AIRDataSet(data_path='./data files/valid data',
                               dim_input=(lstm_input_length, lstm_input_size),
                               dim_output=(lstm_output_length, lstm_output_size))
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_data_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # define Seq2Seq model
    encoder = Encoder(lstm_input_size, hidden_size, device, n_layers=1, dropout=.0)
    decoder = Decoder(lstm_output_size, hidden_size, n_layers=1, dropout=.0)
    model = Act2Act(encoder, decoder, device).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        for train_enc_inputs, train_dec_input, train_dec_outputs in train_data_loader:
            model.zero_grad()

            train_enc_inputs = train_enc_inputs.to(device)
            train_dec_input = train_dec_input.to(device)
            train_dec_outputs = train_dec_outputs.to(device)

            train_predictions = model(train_enc_inputs, train_dec_input, train_dec_outputs,
                                      teacher_forcing_ratio=0.5)

            loss = loss_function(train_predictions, train_dec_outputs)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        # validation
        if epoch % 10 == 0:
            valid_loss = 0.0
            with torch.no_grad():
                for valid_enc_inputs, valid_dec_input, valid_dec_outputs in valid_data_loader:
                    valid_enc_inputs = valid_enc_inputs.to(device)
                    valid_dec_input = valid_dec_input.to(device)
                    valid_dec_outputs = valid_dec_outputs.to(device)

                    valid_predictions = model(valid_enc_inputs, valid_dec_input, valid_dec_outputs,
                                              teacher_forcing_ratio=.0)

                    loss = loss_function(valid_predictions, valid_dec_outputs)
                    valid_loss += loss.item()

            # print average training/validation loss and accuracy
            print(f"Epoch {epoch}")
            print(f"Training Loss: {train_loss / len(train_data_loader):.5f},",
                  f"Validation Loss: {valid_loss / len(valid_data_loader):.5f}")

            model_path = f'./models/model_{epoch:04d}.pth'
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
