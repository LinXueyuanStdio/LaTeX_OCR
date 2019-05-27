import time
import math

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import tensorflow as tf
import numpy as np


def getWH(img_w, img_h):
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w - 2), np.ceil(img_h - 2)
    return int(img_w), int(img_h)

class EncoderCNN(nn.Module):
    def __init__(self, config, training=False):
        super(EncoderCNN, self).__init__()
        self._config = config
        self.cnn = self.getCNN(self._config.encoder_cnn)

    def getCNN(self, cnn_name):
        if cnn_name == "vanilla":
            return nn.Sequential(
                # conv + max pool -> /2
                # 64 个 3*3 filters, strike = (1, 1), output_img.shape = ceil(L/S) = ceil(input/strike) = (H, W)
                nn.Conv2d(in_channels=3, out_channels=64,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # conv + max pool -> /2
                nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # regular conv -> id
                nn.Conv2d(in_channels=128, out_channels=256,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                nn.Conv2d(in_channels=256, out_channels=512,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                # conv
                nn.Conv2d(in_channels=512, out_channels=512,  kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
            )
        elif cnn_name == "cnn":
            return nn.Sequential(
                # conv + max pool -> /2
                # 64 个 3*3 filters, strike = (1, 1), output_img.shape = ceil(L/S) = ceil(input/strike) = (H, W)
                nn.Conv2d(in_channels=3, out_channels=64,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # conv + max pool -> /2
                nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # regular conv -> id
                nn.Conv2d(in_channels=128, out_channels=256,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                nn.Conv2d(in_channels=256, out_channels=512,  kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                nn.Conv2d(in_channels=512, out_channels=512,  kernel_size=(2, 4), stride=2, padding=1),
                nn.ReLU(),

                # conv
                nn.Conv2d(in_channels=512, out_channels=512,  kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
            )

    def forward(self, img):
        """
        Args:
            img: [batch, channel, W, H]
        return:
            out: [batch, 512, W/2/2/2-2, H/2/2/2-2]
        """
        out = self.cnn(img)
        if self._config.positional_embeddings:
            # positional embeddings
            out = self.add_timing_signal_nd_torch(out)
        return out

    def add_timing_signal_nd_torch(x, min_timescale=1.0, max_timescale=1.0e4):
        """嵌入位置信息（positional）
        在Tensor中添加一堆不同频率的正弦曲线。即给输入张量的每个 channels 在对应 positional 维度中加上不同频率和相位的正弦曲线。

        这可以让注意力层学习到绝对和相对位置。
        将信号添加到 query 和 memory 输入当成注意力。
        使用相对位置是可能的，因为 sin(a + b) 和 cos(a + b) 可以用 b，sin(a)和cos(a) 表示。也就是换个基。

        x 是有 n 个 “positional” 维的张量，例如序列是一维，一个 positional 维；图像是二维，两个 positional 维

        我们使用从 min_timescale 开始到 max_timescale 结束的 timescales 序列。不同 timescales 的数量等于 channels//（n * 2）。
        对于每个 timescales，我们生成两个正弦信号 sin(timestep/timescale) 和 cos(timestep/timescale)。
        在 channels 维上将这些正弦曲线连接起来。
        """
        static_shape = x.shape  # [20, 512, 14, 14]
        num_dims = len(static_shape) - 2  # 2
        channels = static_shape[1]  # 512
        num_timescales = channels // (num_dims * 2)  # 128
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1)
        )  # 0.1
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales).float()
                                                   * (-log_timescale_increment))  # len == 128
        for dim in range(num_dims):  # dim == 0; 1
            length = static_shape[dim + 2]  # 14
            position = torch.arange(length).float()  # len == 14
            # inv = [128, 1]， pos = [1, 14], scaled_time = [128, 14]
            scaled_time = inv_timescales.unsqueeze(1) * position.unsqueeze(0)
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=0)  # [256， 14]

            prepad = dim * 2 * num_timescales  # 0; 256
            postpad = channels - (dim + 1) * 2 * num_timescales  # 256; 0

            signal = F.pad(signal, (0, 0, prepad, postpad))  # [512, 14]

            signal = signal.unsqueeze(0)
            for _ in range(dim):
                signal = signal.unsqueeze(2)  # [512, 14]
            for _ in range(num_dims - 1 - dim):
                signal = signal.unsqueeze(-1)
            x += signal  # [1, 512, 14, 1]; [1, 512, 1, 14]


class DecoderRNN(nn.Module):
    def __init__(self, config, n_tok, id_end):
        super(DecoderRNN, self).__init__()
        self._config = config
        self._n_tok = n_tok
        self._id_end = id_end
        self._tiles = 1 if config.decoding == "greedy" else config.beam_size

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length



    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class Img2Seq(nn.Module):
    def __init__(self, config):
        super(Img2Seq, self).__init__()
        self._config = config
        self.encoder = EncoderCNN(config)
        self.decoder = DecoderRNN(config)

    def forward(self):
        pass

    def parameters(self):
        return list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(self.encoder.bn.parameters())
