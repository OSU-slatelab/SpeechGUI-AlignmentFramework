import torch
from torch import nn
from conformer import ConformerBlock
from transformers import BertModel
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

BERTMODEL = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
device = torch.device('cpu')
BERTMODEL = BERTMODEL.to(device)
BERTMODEL.eval()


class SpeechClassifierModel(nn.Module):
    def __init__(self, num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional, device):
        super(SpeechClassifierModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.direction = 2 if bidirectional else 1
        self.device = device
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=True)

        self.fc = nn.Linear(hidden_size*self.direction, num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        out = nn.AdaptiveAvgPool1d(1)(out.permute(0, 2, 1))
        out = torch.sigmoid(out)
        return out


class ConformerModel(nn.Module):
    def __init__(self, num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional, device):
        super(ConformerModel, self).__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.direction = 2 if bidirectional else 1
        self.device = device
        self.num_blocks = 5
        self.projTr = nn.Linear(768, 256)
        self.projSr = nn.Linear(40, 256)
        self.conformer_blocks = nn.ModuleList([ConformerBlock(dim=256,
                                                              dim_head=64,
                                                              heads=4,
                                                              ff_mult=4,
                                                              conv_expansion_factor=2,
                                                              conv_kernel_size=31,
                                                              attn_dropout=0.1,
                                                              ff_dropout=0.1,
                                                              conv_dropout=0.1)
                                              for _ in range(self.num_blocks)])

        self.fc1 = nn.Linear(hidden_size*self.direction, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size*self.direction, num_classes)

    def forward(self, logmels, bEmb):
        unpacked_logmels, lengths = pad_packed_sequence(logmels, batch_first=True)
        unpacked_text, textlengths = pad_packed_sequence(bEmb, batch_first=True)
        batch_size = unpacked_logmels.size(0)
        out_features = []
        for i in range(batch_size):
            logmel = unpacked_logmels[i][:lengths[i]]
            logmel = self.projSr(logmel)
            textToken = unpacked_text[i][:textlengths[i]]
            with torch.no_grad():
                textEmb = BERTMODEL(textToken.unsqueeze(0)).hidden_states[0]
            textEmb = self.projTr(textEmb.squeeze(0))
            in_features = torch.cat((logmel, textEmb), dim=0)
            out_features.append(in_features)

        out_features = pack_sequence(out_features, enforce_sorted=False)
        out_features, outLengths = pad_packed_sequence(out_features, batch_first=True)
        for block in self.conformer_blocks:
            out_features = block(out_features)
        out = nn.AdaptiveAvgPool1d(1)(out_features.permute(0, 2, 1))
        out = self.fc1(out.permute(0, 2, 1))
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.squeeze()
        return out
