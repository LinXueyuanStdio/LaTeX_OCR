import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

# UniChannel ConvNet
class ConvNet(nn.Module):
    def __init__(self, w, h, rnn_hidden_sz):
        super(ConvNet, self).__init__()
        self.w = w
        self.h = h
        self.fc1w = int(((w - 4)/2 - 4)/2)
        self.fc1h = int(((h - 4)/2 - 4)/2)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * self.fc1w * self.fc1h, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, rnn_hidden_sz)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Encoder(nn.Module):
    def __init__(self, one_hot_sz, hidden_sz, n_layers):
        super(Encoder).__init__()
        self.embedding = nn.Embedding(one_hot_sz, hidden_sz)
        self.gru = nn.GRU(one_hot_sz, one_hot_sz, n_layers,
                          dropout=(0 if n_layers <= 1 else 1), bidirectional=True)

    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_sz, output_sz):
        super(Decoder, self).__init__()
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.embedding = nn.Embedding(output_sz, hidden_sz)
        self.gru = nn.GRU(hidden_sz, hidden_sz)
        self.out = nn.Linear(hidden_sz, output_sz)

    def forward(self, input, hidden):
        # one step input, of shape (batch_size, 1),
        # hence embedding outputs of shape (batch_size, 1, hidden_sz)
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = F.softmax(output, dim=2)
        return output, hidden


def oh2ix(oh_vec):
    return 0
def word2ix(word, dict):
    return 0
'''
self.encode = nn.GRU(rnn_in_sz, rnn_in_sz, en_layers,
                          dropout=(0 if en_layers <= 0 else 1), bidirectional=True)
        self.decode = nn.GRU(rnn_in_sz, rnn_in_sz, dn_layers)
        
        def encoder(self, x, hidden):
        outputs, hidden = self.encode(x, hidden)
        outputs = outputs[:, :, :self.rnn_in_sz] + outputs[:, :, self.rnn_in_sz:]
        return outputs, hidden
        '''