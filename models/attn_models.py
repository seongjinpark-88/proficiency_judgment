import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
reference: https://github.com/mttk/rnn-classifier/blob/master/model.py
'''


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gru_layers, dropout, bidirectional=True):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.hidden_dim,
                          num_layers=self.num_gru_layers,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=self.bidirectional)

    def forward(self, acoustic_input):
        return self.rnn(acoustic_input)


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1)  # [BxQ] -> [Bx1xQ]
        # print("query size (B1Q): ", query.size())
        keys = keys.transpose(1, 2)  # [TxBxK] -> [BxKxT]
        # print("key size (BKT): ", keys.size())
        energy = torch.bmm(query, keys)  # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        # print("energy size (B1T): ", energy.size())
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize
        # print("energy size after softmax: ", energy.size())
        # values = values.transpose(0, 1)  # [TxBxV] -> [BxTxV]
        # print("value size after transpose: ", values.size())
        linear_combination = torch.bmm(energy, values).squeeze(1)  # [Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination


class AcousticAttn(nn.Module):
    def __init__(self, encoder, attention):
        super(AcousticAttn, self).__init__()
        self.encoder = encoder
        self.attention = attention
        # self.decoder_1 = nn.Linear(hidden_dim, 256)
        # self.decoder_2 = nn.Linear(256, 128)
        # self.final_decoder = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, acoustic_input, length_input):
        # packed_input = nn.utils.rnn.pack_padded_sequence(
        #     acoustic_input, length_input, batch_first=True, enforce_sorted=False
        # )
        # outputs, hidden = self.encoder(packed_input)
        # acoustic_input = acoustic_input.transpose(1, 2)
        acoustic_input = self.dropout(acoustic_input)
        outputs, hidden = self.encoder(acoustic_input)
        # print("outputs: ", outputs.size())
        # print("hidden from encoder: ", hidden.size())
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[1]  # take the cell state

        if self.encoder.bidirectional:  # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        # max across T?
        # Other options (work worse on a few tests):
        # linear_combination, _ = torch.max(outputs, 0)
        # linear_combination = torch.mean(outputs, 0)
        # print("hidden after cat: ", hidden.size())
        energy, linear_combination = self.attention(hidden, outputs, outputs)
        # print("linear: ", linear_combination.size())
        # logits = F.relu(self.decoder_1(linear_combination))
        # logits = F.relu(self.decoder_2(logits))
        # logits = self.final_decoder(logits)
        # logits = self.decoder(linear_combination)
        # return logits, energy
        return linear_combination, energy
