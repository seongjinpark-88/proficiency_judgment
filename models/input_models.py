import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.svm import SVR

from models.attn_models import *


class MTLModels(nn.Module):

    def __init__(self, params):
        super(MTLModels, self).__init__()

        self.GRU_Raw = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=False
        )
        self.raw_linear = nn.Linear(params.acoustic_gru_hidden_dim, params.fc_hidden_dim)

        self.GRU_Phon = nn.LSTM(
            input_size=params.phon_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=False
        )
        self.phon_linear = nn.Linear(params.acoustic_gru_hidden_dim, params.fc_hidden_dim)

        self.MLP_Phono = nn.Linear(params.phono_dim, params.fc_hidden_dim)

        self.fc_acc_1 = nn.Linear(params.fc_hidden_dim * 3, 256)
        self.fc_acc_2 = nn.Linear(256, 1)
        self.fc_flu_1 = nn.Linear(params.fc_hidden_dim * 3, 256)
        self.fc_flu_2 = nn.Linear(256, 1)
        self.fc_com_1 = nn.Linear(params.fc_hidden_dim * 3, 256)
        self.fc_com_2 = nn.Linear(256, 1)

    def forward(self, input_features):
        raw_input = input_features[0]
        raw_input_len = input_features[1]
        phon_input = input_features[2]
        phon_input_len = input_features[3]
        phono_input = input_features[4]

        # print(raw_input.size(), phon_input.size(), phono_input.size())
        raw_packed = nn.utils.rnn.pack_padded_sequence(
            raw_input, raw_input_len, batch_first=True, enforce_sorted=False
        )

        phon_packed = nn.utils.rnn.pack_padded_sequence(
            phon_input, phon_input_len, batch_first=True, enforce_sorted=False
        )

        raw_output, (hidden_raw, cell_raw) = self.GRU_Raw(raw_packed)
        encoded_raw = F.dropout(hidden_raw[-1], 0.3)
        raw_intermediate = torch.tanh(F.dropout(self.raw_linear(encoded_raw), 0.5))

        phon_output, (hidden_phon, cell_phon) = self.GRU_Phon(phon_packed)
        encoded_phon = F.dropout(hidden_phon[-1], 0.3)
        phon_intermediate = torch.tanh(F.dropout(self.phon_linear(encoded_phon), 0.5))

        # phono_output = self.MLP_Phono(phono_input)
        phono_intermediate = torch.relu(F.dropout(self.MLP_Phono(phono_input), 0.5))

        concat_output = torch.cat((raw_intermediate, phon_intermediate, phono_intermediate), 1)

        self.acc_out = self.fc_acc_2(self.fc_acc_1(concat_output))
        self.flu_out = self.fc_flu_2(self.fc_flu_1(concat_output))
        self.com_out = self.fc_com_2(self.fc_com_1(concat_output))
        return [self.acc_out, self.flu_out, self.com_out]


class MTLLossWrapper(nn.Module):
    def __init__(self, task_num, model):
        super(MTLLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, input_feat, acc, flu, comp):
        mse = nn.MSELoss(reduction='mean')

        preds = self.model(input_feat, input_type="phonological")

        loss0 = mse(preds[0].squeeze(1), acc)
        loss1 = mse(preds[1].squeeze(1), flu)
        loss2 = mse(preds[2].squeeze(1), comp)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * loss1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2 * loss2 + self.log_vars[2]

        return loss0 + loss1 + loss2, self.log_vars.data.tolist()


class AcousticRNNmtl(nn.Module):

    def __init__(self, params):
        super(AcousticRNNmtl, self).__init__()
        self.audio_dim = params.audio_dim
        self.hidden_dim = params.acoustic_gru_hidden_dim
        self.num_layers = params.num_gru_layers

        self.GRULayer = nn.GRU(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=False
        )

        self.linear = nn.Linear(params.acoustic_gru_hidden_dim, params.fc_hidden_dim)

        self.fc_acc_1 = nn.Linear(params.fc_hidden_dim, 256)
        self.fc_acc_2 = nn.Linear(256, 128)
        self.fc_acc = nn.Linear(128, 1)

        self.fc_flu_1 = nn.Linear(params.fc_hidden_dim, 256)
        self.fc_flu_2 = nn.Linear(256, 128)
        self.fc_flu = nn.Linear(128, 1)

        self.fc_com_1 = nn.Linear(params.fc_hidden_dim, 256)
        self.fc_com_2 = nn.Linear(256, 128)
        self.fc_com = nn.Linear(128, 1)

    def forward(self, input_features, input_length):
        input_feat = input_features
        input_len = input_length
        packed = nn.utils.rnn.pack_padded_sequence(
            input_feat, input_len, batch_first=True, enforce_sorted=False
        )

        outputs, hidden = self.GRULayer(packed)

        encoded = F.dropout(hidden[-1], 0.3)
        shared_layer = torch.tanh(F.dropout(self.linear(encoded), 0.3))

        self.acc_inter = torch.tanh(self.fc_acc_2(F.dropout(torch.tanh(self.fc_acc_1(shared_layer)), 0.2)))
        self.flu_inter = torch.tanh(self.fc_flu_2(F.dropout(torch.tanh(self.fc_flu_1(shared_layer)), 0.2)))
        self.com_inter = torch.tanh(self.fc_com_2(F.dropout(torch.tanh(self.fc_com_1(shared_layer)), 0.2)))

        self.acc_out = self.fc_acc(F.dropout(self.acc_inter, 0.2))
        self.flu_out = self.fc_flu(F.dropout(self.flu_inter, 0.2))
        self.com_out = self.fc_com(F.dropout(self.com_inter, 0.2))

        return [self.acc_out, self.flu_out, self.com_out]


class AcousticRNN(nn.Module):

    def __init__(self, params):
        super(AcousticRNN, self).__init__()

        self.audio_dim = params.audio_dim
        self.hidden_dim = params.acoustic_gru_hidden_dim
        self.num_layers = params.num_gru_layers

        self.GRULayer = nn.GRU(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=False
        )

        self.linear = nn.Linear(params.acoustic_gru_hidden_dim, params.fc_hidden_dim)
        self.fc_1 = nn.Linear(params.fc_hidden_dim, 256)
        self.fc_2 = nn.Linear(256, 1)

    def forward(self, input_features, input_length):
        input_features = input_features.squeeze(2)
        input_features = input_features.transpose(1, 2)
        # print(input_features.size())
        packed_feats = nn.utils.rnn.pack_padded_sequence(
            input_features, input_length, batch_first=True, enforce_sorted=False
        )

        outputs, hidden = self.GRULayer(packed_feats)
        encoded = F.dropout(torch.tanh(hidden[-1]), 0.3)
        intermediate = F.dropout(torch.tanh(self.linear(encoded)), 0.3)

        self.prediction = self.fc_2(F.dropout(torch.tanh(self.fc_1(intermediate)), 0.2))


        return self.prediction


class MultiAcousticModelEarlyMTL(nn.Module):
    def __init__(self, params):
        super(MultiAcousticModelEarlyMTL, self).__init__()
        # input dimensions
        self.attn_dim = params.attn_dim

        self.audio_dim = params.audio_dim

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=params.bidirectional
        )

        encoder = Encoder(input_dim=params.attn_dim,
                          hidden_dim=params.acoustic_gru_hidden_dim,
                          num_gru_layers=params.num_gru_layers,
                          dropout=params.dropout,
                          bidirectional=params.bidirectional)

        attention_dim = params.acoustic_gru_hidden_dim if not params.bidirectional else 2 * params.acoustic_gru_hidden_dim
        attention = Attention(attention_dim, attention_dim, attention_dim)

        self.acoustic_model = AcousticAttn(
            encoder=encoder,
            attention=attention
        )

        if params.bidirectional:
            self.audio_hidden_dim = params.acoustic_gru_hidden_dim
            self.attn_hidden_dim = params.acoustic_gru_hidden_dim * 2
            self.fc_input_dim = self.audio_hidden_dim + self.attn_hidden_dim
        else:
            self.fc_input_dim = 2 * params.acoustic_gru_hidden_dim

        self.fc1_acc = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2_acc = nn.Linear(params.fc_hidden_dim, 64)
        self.fc3_acc = nn.Linear(64, params.output_dim)

        self.fc1_flu = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2_flu = nn.Linear(params.fc_hidden_dim, 64)
        self.fc3_flu = nn.Linear(64, params.output_dim)

        self.fc1_flu = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2_flu = nn.Linear(params.fc_hidden_dim, 64)
        self.fc3_flu = nn.Linear(64, params.output_dim)

    def forward(self,
                audio_input,
                audio_length,
                acoustic_input,
                acoustic_length):

        audio_input = audio_input.transpose(1, 2)
        attn_output, _ = self.acoustic_model(audio_input, audio_length)
        # print("before: ", acoustic_input.size())
        acoustic_input = torch.squeeze(acoustic_input, 1).transpose(1, 2)
        # print("after: ", acoustic_input.size())
        packed = nn.utils.rnn.pack_padded_sequence(
            acoustic_input, acoustic_length, batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.acoustic_rnn(packed)

        rnn_output = F.dropout(hidden[-1], 0.3)

        # print("attn size: ", attn_output.size())
        # print("rnn size: ", rnn_output.size())

        inputs = torch.cat((attn_output, rnn_output), 1)
        # print("cat. input size: ", inputs.size())

        acc_output = torch.tanh(F.dropout(self.fc1_acc(inputs), 0.3))
        acc_output = torch.tanh(F.dropout(self.fc2_acc(acc_output), 0.3))
        acc_output = self.fc3_acc(acc_output)

        flu_output = torch.tanh(F.dropout(self.fc1_acc(inputs), 0.3))
        flu_output = torch.tanh(F.dropout(self.fc2_acc(flu_output), 0.3))
        flu_output = self.fc3_flu(flu_output)

        comp_output = torch.tanh(F.dropout(self.fc1_acc(inputs), 0.3))
        comp_output = torch.tanh(F.dropout(self.fc2_acc(comp_output), 0.3))
        comp_output = self.fc3_acc(comp_output)

        return acc_output, flu_output, comp_output


class SimpleFFN(nn.Module):

    def __init__(self, params):
        super(SimpleFFN, self).__init__()
        self.dropout = params.dropout
        # self.num_layers = params.num_fc_layers
        self.linear = nn.Linear(params.phono_dim, params.fc_hidden_dim_phon)
        self.fc_1 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        self.fc_2 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        self.fc_3 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        # self.fc_4 = nn.Linear(params.fc_hidden_dim_phon, 1)
        self.fc_4 = nn.Linear(params.fc_hidden_dim_phon, 1)
        # self.fc_5 = nn.Linear(params.fc_hidden_dim_phon, 1)

    def forward(self, input_features):
        # print(input_features.size())
        encoded = F.dropout(torch.tanh(self.linear(input_features)), self.dropout)
        fcn1 = F.dropout(torch.tanh(self.fc_1(encoded)), self.dropout)
        fcn2 = F.dropout(torch.tanh(self.fc_2(fcn1)), self.dropout)
        fcn3 = F.dropout(torch.tanh(self.fc_2(fcn2)), self.dropout)
        # fcn4 = F.dropout(torch.tanh(self.fc_4(fcn3)), self.dropout)
        self.prediction = self.fc_4(fcn3)

        return self.prediction


class SimpleFFNmtl(nn.Module):

    def __init__(self, params):
        super(SimpleFFNmtl, self).__init__()

        # self.num_layers = params.num_fc_layers

        self.dropout = params.dropout
        self.linear = nn.Linear(params.phono_dim, params.fc_hidden_dim_phon)
        self.fc_1 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        self.fc_2 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        self.fc_3 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)


        self.fc_acc_1 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        self.fc_acc_2 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        self.fc_acc = nn.Linear(params.fc_hidden_dim_phon, 1)

        self.fc_flu_1 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        self.fc_flu_2 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        self.fc_flu = nn.Linear(params.fc_hidden_dim_phon, 1)

        self.fc_com_1 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        self.fc_com_2 = nn.Linear(params.fc_hidden_dim_phon, params.fc_hidden_dim_phon)
        self.fc_com = nn.Linear(params.fc_hidden_dim_phon, 1)

    def forward(self, input_features):
        shared_layer = F.dropout(torch.tanh(self.linear(input_features)), self.dropout)
        shared_layer = F.dropout(torch.tanh(self.fc_1(shared_layer)), self.dropout)
        shared_layer = F.dropout(torch.tanh(self.fc_2(shared_layer)), self.dropout)
        shared_layer = F.dropout(torch.tanh(self.fc_3(shared_layer)), self.dropout)

        self.acc_inter = self.fc_acc_2(F.dropout(torch.tanh(self.fc_acc_1(shared_layer)), self.dropout))
        self.flu_inter = self.fc_flu_2(F.dropout(torch.tanh(self.fc_flu_1(shared_layer)), self.dropout))
        self.com_inter = self.fc_com_2(F.dropout(torch.tanh(self.fc_com_1(shared_layer)), self.dropout))

        self.acc_out = self.fc_acc(F.dropout(self.acc_inter, self.dropout))
        self.flu_out = self.fc_flu(F.dropout(self.flu_inter, self.dropout))
        self.com_out = self.fc_com(F.dropout(self.com_inter, self.dropout))

        # self.acc_out = self.fc_acc(F.dropout(torch.tanh(shared_layer), self.dropout))
        # self.flu_out = self.fc_flu(F.dropout(torch.tanh(shared_layer), self.dropout))
        # self.com_out = self.fc_com(F.dropout(torch.tanh(shared_layer), self.dropout))

        return [self.acc_out, self.flu_out, self.com_out]

class AcousticFFN(nn.Module):

    def __init__(self, params):
        super(AcousticFFN, self).__init__()

        self.dropout = params.dropout
        self.linear = nn.Linear(params.acoustic_dim, params.fc_hidden_dim_acoustic)
        self.fc_1 = nn.Linear(params.fc_hidden_dim_acoustic, params.fc_hidden_dim_acoustic)
        self.fc_2 = nn.Linear(params.fc_hidden_dim_acoustic, 64)
        self.fc_3 = nn.Linear(64, 1)

    def forward(self, input_features):
        # print(input_features.size())
        # input_features = torch.mean(input_features, dim=1)
        # print(input_features.size())
        encoded = F.dropout(torch.tanh(self.linear(input_features)), self.dropout)
        fcn1 = F.dropout(torch.tanh(self.fc_1(encoded)), self.dropout)
        fcn2 = F.dropout(torch.tanh(self.fc_2(fcn1)), self.dropout)
        self.prediction = self.fc_3(fcn2)

        return self.prediction


# class AcousticSVR():
#     def __init__(self):
#         self.svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
#
#     def forward(self, input_features, label):
#         return self.svr_rbf.fit(input_features, label).predict(input_features)


class AcousticFFNmtl(nn.Module):

    def __init__(self, params):
        super(AcousticFFNmtl, self).__init__()

        # self.num_layers = params.num_fc_layers
        self.linear = nn.Linear(params.acoustic_dim, params.fc_hidden_dim)
        # self.fc = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)

        self.fc_acc_1 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)
        self.fc_acc_2 = nn.Linear(params.fc_hidden_dim, 64)
        self.fc_acc = nn.Linear(64, 1)

        self.fc_flu_1 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)
        self.fc_flu_2 = nn.Linear(params.fc_hidden_dim, 64)
        self.fc_flu = nn.Linear(64, 1)

        self.fc_com_1 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)
        self.fc_com_2 = nn.Linear(params.fc_hidden_dim, 64)
        self.fc_com = nn.Linear(64, 1)

    def forward(self, input_features):
        # print(input_features.size())
        # input_features = torch.mean(input_features, dim=1)
        # print(input_features.size())

        shared_layer = F.dropout(torch.tanh(self.linear(input_features)), 0.3)
        # shared_layer = F.dropout(torch.tanh(self.fc(shared_layer)), 0.2)

        self.acc_inter = torch.tanh(self.fc_acc_2(F.dropout(torch.tanh(self.fc_acc_1(shared_layer)), 0.3)))
        self.flu_inter = torch.tanh(self.fc_flu_2(F.dropout(torch.tanh(self.fc_flu_1(shared_layer)), 0.3)))
        self.com_inter = torch.tanh(self.fc_com_2(F.dropout(torch.tanh(self.fc_com_1(shared_layer)), 0.3)))

        self.acc_out = self.fc_acc(F.dropout(self.acc_inter, 0.3))
        self.flu_out = self.fc_flu(F.dropout(self.flu_inter, 0.3))
        self.com_out = self.fc_com(F.dropout(self.com_inter, 0.3))

        return [self.acc_out, self.flu_out, self.com_out]


class AcousticSingleAttn(nn.Module):
    def __init__(self, params):
        super(AcousticSingleAttn, self).__init__()
        # input dimensions
        self.attn_dim = params.attn_dim

        self.audio_dim = params.audio_dim

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=params.bidirectional
        )

        encoder = Encoder(input_dim=params.attn_dim,
                          hidden_dim=params.acoustic_gru_hidden_dim,
                          num_gru_layers=params.num_gru_layers,
                          dropout=params.dropout,
                          bidirectional=params.bidirectional)

        attention_dim = params.acoustic_gru_hidden_dim if not params.bidirectional else 2 * params.acoustic_gru_hidden_dim
        attention = Attention(attention_dim, attention_dim, attention_dim)

        self.acoustic_model = AcousticAttn(
            encoder=encoder,
            attention=attention
        )

        if params.bidirectional:
            self.attn_hidden_dim = params.acoustic_gru_hidden_dim * 2
            self.fc_input_dim = self.attn_hidden_dim
        else:
            self.fc_input_dim = 2 * params.acoustic_gru_hidden_dim

        self.fc1_acc = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2_acc = nn.Linear(params.fc_hidden_dim, 64)
        self.fc3_acc = nn.Linear(64, params.output_dim)

    def forward(self,
                input_features,
                input_length):

        attn_output, _ = self.acoustic_model(input_features, input_length)
        # print("before: ", acoustic_input.size())

        inputs = attn_output
        # print("cat. input size: ", inputs.size())

        acc_output = F.dropout(torch.tanh(self.fc1_acc(inputs)), 0.3)
        acc_output = F.dropout(torch.tanh(self.fc2_acc(acc_output)), 0.3)
        acc_output = self.fc3_acc(acc_output)

        return acc_output


class MergeLayer(nn.Module):

    def __init__(self, params):
        super(MergeLayer, self).__init__()

        # self.input_shape = params.hidden_shape

        self.fc_1 = nn.Linear(params.fc_hidden_dim, 256)
        self.fc_2 = nn.Linear(256, 1)

    def forward(self, input_features):
        fcn1 = F.dropout(torch.tanh(self.fc_1(input_features)), 0.2)
        self.prediction = self.fc_2(fcn1)

        return self.prediction


class EmbraceNet(nn.Module):
    def __init__(self,
                 device,
                 input_size_list,
                 params):
        """
        https://github.com/idearibosome/embracenet
        Initialize an EmbraceNet Module
        @param device: torch.device object (cpu / gpu)
        @param input_size_list: list of input sizes [num_feat, shape_input] ("c" in the paper)
        @param embracement_size: the length of the output of the embracement layer
        @param bypass_docking:
            bypass docking step. If True, the shape of input_data should be [batch_size, embracement_size]
        """
        super(EmbraceNet, self).__init__()

        self.device = device
        self.input_size_list = input_size_list
        self.embracement_size = params.embracement_size
        self.bypass_docking = params.bypass_docking
        self.availabilities = params.availabilities
        self.selection_probabilities = params.selection_probabilities

        if not self.bypass_docking:
            for i, input_size in enumerate(input_size_list):
                setattr(self, 'docking_%d' % i, nn.Linear(input_size, self.embracement_size))

    def forward(self, input_list):
        """
        Forward input data to the EmbraceNet module
        @param input_list: A list of input data
        @param availabilities: 2D tensor of shape [batch_size, num_feats],
                               which represents the availability of data for each modality. If None, it assumes that
                               data of all features are available
        @param selection_probabilities: 2D tensor of shape [batch_size, num_feats],
                                      which represents probabilities that output of each docking layer will be
                                      selected ("p" in the paper). If None, same probability will be used.

        @return: 2D tensor of shape [batch_size, embracement_size]
        """

        # check input_data
        assert len(input_list) == len(self.input_size_list)
        num_feats = len(input_list)
        batch_size = input_list[0].shape[0]

        # docking layer
        docking_output_list = []
        if (self.bypass_docking):
            docking_output_list = input_list
        else:
            for i, input_data in enumerate(input_list):
                x = getattr(self, 'docking_%d' % i)(input_data)
                x = nn.functional.relu(x)
                docking_output_list.append(x)

        # check availabilities
        if (self.availabilities is None):
            availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
        else:
            availabilities = self.availabilities.float()

        # adjust selection probabilities
        if (self.selection_probabilities is None):
            selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
        selection_probabilities = torch.mul(selection_probabilities, availabilities)

        probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
        selection_probabilities = torch.div(selection_probabilities, probability_sum)

        # stack docking outputs
        docking_output_stack = torch.stack(docking_output_list,
                                           dim=-1)  # [batch_size, embracement_size, num_modalities]

        # embrace
        feature_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size,
                                            replacement=True)
        feature_toggles = nn.functional.one_hot(feature_indices,
                                                num_classes=num_feats).float()  # [batch_size, embracement_size, num_feat]

        embracement_output_stack = torch.mul(docking_output_stack, feature_toggles)
        embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]

        return embracement_output

class MultiInputNet(nn.Module):
    def __init__(self,
                 device,
                 input_size_list,
                 params):
        """
        https://github.com/idearibosome/embracenet
        Initialize an EmbraceNet Module
        @param device: torch.device object (cpu / gpu)
        @param input_size_list: list of input sizes [num_feat, shape_input] ("c" in the paper)
        @param embracement_size: the length of the output of the embracement layer
        @param bypass_docking:
            bypass docking step. If True, the shape of input_data should be [batch_size, embracement_size]
        """
        super(MultiInputNet, self).__init__()

        self.device = device
        self.input_size_list = input_size_list
        self.embracement_size = params.embracement_size
        self.bypass_docking = params.bypass_docking

        if not self.bypass_docking:
            for i, input_size in enumerate(input_size_list):
                setattr(self, 'docking_%d' % i, nn.Linear(input_size, self.embracement_size))

    def forward(self, input_list):
        """
        Forward input data to the EmbraceNet module
        @param input_list: A list of input data
        @param availabilities: 2D tensor of shape [batch_size, num_feats],
                               which represents the availability of data for each modality. If None, it assumes that
                               data of all features are available
        @param selection_probabilities: 2D tensor of shape [batch_size, num_feats],
                                      which represents probabilities that output of each docking layer will be
                                      selected ("p" in the paper). If None, same probability will be used.

        @return: 2D tensor of shape [batch_size, embracement_size]
        """

        # check input_data
        assert len(input_list) == len(self.input_size_list)
        num_feats = len(input_list)
        batch_size = input_list[0].shape[0]

        # docking layer
        docking_output_list = []
        if (self.bypass_docking):
            docking_output_list = input_list
        else:
            for i, input_data in enumerate(input_list):
                # print(input_data.size())
                x = getattr(self, 'docking_%d' % i)(input_data)
                x = nn.functional.relu(x)
                docking_output_list.append(x)

        # cat docking outputs
        docking_output_cat = torch.cat(docking_output_list, dim=1)

        return docking_output_cat


class MultiInput_single_cv(nn.Module):
    def __init__(self, device, feats, params):
        super(MultiInput_single_cv, self).__init__()
        # input dimensions

        if feats == "AudioAcoustic":
            self.audio_dim = params.audio_dim
            self.acoustic_dim = params.acoustic_dim
            self.num_gru_layers = params.num_gru_layers

            self.AudioGRULayer = nn.GRU(
                input_size=params.audio_dim,
                hidden_size=params.acoustic_gru_hidden_dim,
                num_layers=params.num_gru_layers,
                batch_first=True,
                bidirectional=False
            )
            self.AudioLinear1 = nn.Linear(params.acoustic_gru_hidden_dim, params.fc_hidden_dim)
            self.AudioLinear2 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)

            self.AcousticInput = nn.Linear(params.acoustic_dim, params.fc_hidden_dim)
            self.AcousticLinear1 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)
            self.AcousticLinear2 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)

            input_size = [params.fc_hidden_dim, params.fc_hidden_dim]

        elif feats == "AudioPhon":
            self.audio_dim = params.audio_dim
            self.phono_dim = params.phono_dim
            self.num_gru_layers = params.num_gru_layers

            self.AudioGRULayer = nn.GRU(
                input_size=params.audio_dim,
                hidden_size=params.acoustic_gru_hidden_dim,
                num_layers=params.num_gru_layers,
                batch_first=True,
                bidirectional=False
            )
            self.AudioLinear1 = nn.Linear(params.acoustic_gru_hidden_dim, params.fc_hidden_dim)
            self.AudioLinear2 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)

            self.Linear = nn.Linear(params.phono_dim, params.fc_hidden_dim)
            self.PhonLinear1 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)
            self.PhonLinear2 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)

            input_size = [params.fc_hidden_dim, params.fc_hidden_dim]

        elif feats == "AcousticPhon":
            self.acoustic_dim = params.acoustic_dim
            self.phono_dim = params.phono_dim
            self.num_gru_layers = params.num_gru_layers

            self.AcousticInput = nn.Linear(params.acoustic_dim, params.fc_hidden_dim)
            self.AcousticLinear1 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)
            self.AcousticLinear2 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)

            self.Linear = nn.Linear(params.phono_dim, params.fc_hidden_dim)
            self.PhonLinear1 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)
            self.PhonLinear2 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)

            input_size = [params.fc_hidden_dim, params.fc_hidden_dim]

        elif feats == "All":
            self.audio_dim = params.audio_dim
            self.acoustic_dim = params.acoustic_dim
            self.phono_dim = params.phono_dim

            self.AudioGRULayer = nn.GRU(
                input_size=params.audio_dim,
                hidden_size=params.acoustic_gru_hidden_dim,
                num_layers=params.num_gru_layers,
                batch_first=True,
                bidirectional=False
            )

            self.AudioLinear1 = nn.Linear(params.acoustic_gru_hidden_dim, params.fc_hidden_dim)
            self.AudioLinear2 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)

            self.AcousticInput = nn.Linear(params.acoustic_dim, params.fc_hidden_dim)
            self.AcousticLinear1 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)
            self.AcousticLinear2 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)

            self.Linear = nn.Linear(params.phono_dim, params.fc_hidden_dim)
            self.PhonLinear1 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)
            self.PhonLinear2 = nn.Linear(params.fc_hidden_dim, params.fc_hidden_dim)

            input_size = [params.fc_hidden_dim, params.fc_hidden_dim, params.fc_hidden_dim]

        self.multiinputnet = MultiInputNet(device=device, input_size_list=input_size, params=params)

        self.input_dimension = sum(input_size)

        self.fc1 = nn.Linear(self.input_dimension, params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self,
                feats,
                input_features,
                input_lengths):

        if feats == "AudioAcoustic":
            audio_feats = input_features[0]
            audio_length = input_lengths
            acoustic_feats = input_features[1].squeeze(1)


            audio_packed_feats = nn.utils.rnn.pack_padded_sequence(
                audio_feats, audio_length, batch_first=True, enforce_sorted=False
            )

            audio_outputs, audio_hidden = self.AudioGRULayer(audio_packed_feats)
            audio_encoded = F.dropout(audio_hidden[-1], 0.2)
            audio_layer_inter = F.dropout(torch.tanh(self.AudioLinear1(audio_encoded)), 0.2)
            audio_layer = F.dropout(torch.tanh(self.AudioLinear2(audio_layer_inter)), 0.2)

            acoustic_encoded = F.dropout(torch.tanh(self.AcousticInput(acoustic_feats)), 0.2)
            acoustic_layer_inter = F.dropout(torch.tanh(self.AcousticLinear1(acoustic_encoded)), 0.2)
            acoustic_layer = F.dropout(torch.tanh(self.AcousticLinear2(acoustic_layer_inter)), 0.2)

            multiinputnet_output = self.multiinputnet([audio_layer, acoustic_layer])

            output1 = F.dropout(torch.relu(self.fc1(multiinputnet_output)), 0.2)
            output2 = F.dropout(torch.relu(self.fc2(output1)), 0.2)
            output = self.fc3(output2)

        elif feats == "AudioPhon":
            audio_feats = input_features[0]
            audio_length = input_lengths
            phon_feats = input_features[1]

            audio_packed_feats = nn.utils.rnn.pack_padded_sequence(
                audio_feats, audio_length, batch_first=True, enforce_sorted=False
            )

            audio_outputs, audio_hidden = self.AudioGRULayer(audio_packed_feats)

            audio_encoded = F.dropout(audio_hidden[-1], 0.2)
            audio_layer_inter = F.dropout(torch.tanh(self.AudioLinear1(audio_encoded)), 0.2)
            audio_layer = F.dropout(torch.tanh(self.AudioLinear2(audio_layer_inter)), 0.2)

            phon_encoded = F.dropout(torch.tanh(self.Linear(phon_feats)), 0.2)
            phon_layer_inter = F.dropout(torch.tanh(self.PhonLinear1(phon_encoded)), 0.2)
            phon_layer = F.dropout(torch.tanh(self.PhonLinear2(phon_layer_inter)), 0.2)

            multiinputnet_output = self.multiinputnet([audio_layer, phon_layer])

            output1 = F.dropout(torch.relu(self.fc1(multiinputnet_output)), 0.2)
            output2 = F.dropout(torch.relu(self.fc2(output1)), 0.2)
            output = self.fc3(output2)

        elif feats == "AcousticPhon":
            acoustic_feats = input_features[0].squeeze(1)
            phon_feats = input_features[1]

            acoustic_encoded = F.dropout(torch.tanh(self.AcousticInput(acoustic_feats)), 0.2)
            acoustic_layer_inter = F.dropout(torch.tanh(self.AcousticLinear1(acoustic_encoded)), 0.2)
            acoustic_layer = F.dropout(torch.tanh(self.AcousticLinear2(acoustic_layer_inter)), 0.2)

            phon_encoded = F.dropout(torch.tanh(self.Linear(phon_feats)), 0.2)
            phon_layer_inter = F.dropout(torch.tanh(self.PhonLinear1(phon_encoded)), 0.2)
            phon_layer = F.dropout(torch.tanh(self.PhonLinear2(phon_layer_inter)), 0.2)

            multiinputnet_output = self.multiinputnet([acoustic_layer, phon_layer])

            output1 = F.dropout(torch.relu(self.fc1(multiinputnet_output)), 0.2)
            output2 = F.dropout(torch.relu(self.fc2(output1)), 0.2)
            output = self.fc3(output2)

        elif feats == "All":
            audio_feats = input_features[0]
            audio_length = input_lengths
            acoustic_feats = input_features[1].squeeze(1)
            phon_feats = input_features[2]

            audio_packed_feats = nn.utils.rnn.pack_padded_sequence(
                audio_feats, audio_length, batch_first=True, enforce_sorted=False
            )

            audio_outputs, audio_hidden = self.AudioGRULayer(audio_packed_feats)

            audio_encoded = F.dropout(audio_hidden[-1], 0.2)
            audio_layer_inter = F.dropout(torch.tanh(self.AudioLinear1(audio_encoded)), 0.2)
            audio_layer = F.dropout(torch.tanh(self.AudioLinear2(audio_layer_inter)), 0.2)

            acoustic_encoded = F.dropout(torch.tanh(self.AcousticInput(acoustic_feats)), 0.2)
            acoustic_layer_inter = F.dropout(torch.tanh(self.AcousticLinear1(acoustic_encoded)), 0.2)
            acoustic_layer = F.dropout(torch.tanh(self.AcousticLinear1(acoustic_layer_inter)), 0.2)

            phon_encoded = F.dropout(torch.tanh(self.Linear(phon_feats)), 0.2)
            phon_layer_inter = F.dropout(torch.tanh(self.PhonLinear1(phon_encoded)), 0.2)
            phon_layer = F.dropout(torch.tanh(self.PhonLinear2(phon_layer_inter)), 0.2)

            multiinputnet_output = self.multiinputnet([audio_layer, acoustic_layer, phon_layer])

            output1 = F.dropout(torch.relu(self.fc1(multiinputnet_output)), 0.2)
            output2 = F.dropout(torch.relu(self.fc2(output1)), 0.2)
            output = self.fc3(output2)

        return output
