"""
LSTMCell
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        # 初始化输入大小
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x, hx, cx):
        # x: 100,128
        # hx: 100,128
        # cx: 100,128
        gates = self.x2h(x) + self.h2h(hx)
        # gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))
        return hy, cy


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = LSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim))

        # 初始化cell_state
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(x.size(0), self.hidden_dim))

        outs = []
        cn = c0  # 100,128
        hn = h0  # 100,128

        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:, seq, :], hn, cn)
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out
