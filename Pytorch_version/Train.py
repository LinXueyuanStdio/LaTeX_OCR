import Net
import ImgData as ID
import torch.nn as nn
import torch as t
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

shape = (320, 80)
def build_formula_sym(formula_dir):
    f = open(formula_dir, 'r', encoding='UTF-8')
    sym_dict = ID.FormulaSym()
    for line in f:
        sym_dict.add_formula(line)
    return sym_dict

# input tensor from SOS to the last char,
# shape of (input_length, batch_size), each element is the index of the symbol
def build_input(line, sym_dict):
    tensor = t.zeros(len(line) + 1, 1)
    tensor[0][0] = 0 # 0 is SOS
    for i in range(len(line)):
        tensor[i + 1][0] = sym_dict[line[i]]
    tensor = tensor.long()
    return tensor

# shift the input, starts with the first symbol, ends with EOS
def build_target(line, sym_dict):
    tensor = t.zeros(len(line) + 1, 1)
    for i in range(len(line)):
        tensor[i][0] = sym_dict[line[i]]
    tensor[-1][0] = 1
    tensor = tensor.long()
    return tensor

def prepare_rnn_pair(line, sym_dict):
    input = build_input(line, sym_dict)
    target = build_target(line, sym_dict)
    return [input, target]

def build_network(conv_input_shape, rnn_hidden_sz, n_ch):
    conv = Net.ConvNet(conv_input_shape[0], conv_input_shape[1], rnn_hidden_sz)
    decoder = Net.Decoder(rnn_hidden_sz, n_ch)
    conv_optim = optim.SGD(conv.parameters(), lr=0.001)
    decoder_optim = optim.SGD(decoder.parameters(), lr=0.001)
    return conv, decoder, conv_optim, decoder_optim

def train(latex_nth2img, latex_formulas, img_dir, max_len, teacher_forced=0.5):
    trainset = ID.SmallImgDataSet(max_len, latex_nth2img,
                                  latex_formulas, img_dir,
                                  transform=transforms.Compose([
                                      ID.UniChannel(),
                                      ID.Rescale(shape),
                                      ID.ToTensor()
                                  ]))
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    sym_dict = build_formula_sym(latex_formulas)
    conv, decoder, conv_optim, decoder_optim = build_network(shape, 24, len(sym_dict))
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            input = data['image']
            latex = data['latex'][0]
            pair = prepare_rnn_pair(latex, sym_dict)

            conv_optim.zero_grad()
            decoder_optim.zero_grad()
            input = input.float()
            conv_out = conv(input)
            next_hidden = conv_out.view(1, 1, -1)
            decoder_out = pair[0][0]

            loss = 0
            for j in range(len(pair[0])):
                teacher = np.random.rand() < teacher_forced
                if j == 0 or teacher:
                    decoder_out, next_hidden = decoder(pair[0][j], next_hidden)
                else:
                    decoder_out, next_hidden = decoder(decoder_out, next_hidden)
                decoder_out = decoder_out.view(1, -1)
                l = criterion(decoder_out, pair[1][j])
                loss += l
                running_loss += l.item()
                _, decoder_out = t.max(decoder_out, 1)
            if i % 500 == 499:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/500))
                running_loss = 0.0
            loss.backward()
            conv_optim.step()
            decoder_optim.step()




