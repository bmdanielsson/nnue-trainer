import torch
import struct

from torch import nn


# Number of inputs
NUM_SQ = 64
NUM_PT = 12
NUM_INPUTS = NUM_SQ*NUM_PT

# 3 layer fully connected network
L1 = 256
L2 = 8
L3 = 16

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.input = nn.Linear(NUM_INPUTS, L1)
        self.l1 = nn.Linear(2 * L1, L2)
        self.l2 = nn.Linear(L2, L3)
        self.output = nn.Linear(L3, 1)


    def forward(self, us, them, w_in, b_in):
        w = self.input(w_in)
        b = self.input(b_in)
        l0_ = (us*torch.cat([w, b], dim=1)) + (them*torch.cat([b, w], dim=1))
        l0_ = torch.clamp(l0_, 0.0, 1.0)
        l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
        l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
        x = self.output(l2_)
        return x


    def clamp_weights(self):
        # L1
        data = self.l1.weight.data
        data.clamp(-127.0/64.0, 127.0/64.0)
        self.l1.weight.data.copy_(data)

        # L2
        data = self.l2.weight.data
        data.clamp(-127.0/64.0, 127.0/64.0)
        self.l2.weight.data.copy_(data)

        # Output
        data = self.output.weight.data
        data.clamp(-127.0*127.0/64.0, 127.0*127.0/64.0)
        self.output.weight.data.copy_(data)


def loss_function(wdl, pred, batch):
    us, them, white, black, outcome, score = batch
    
    wdl_eval_model = (pred*600.0/361).sigmoid()
    wdl_eval_target = (score/410).sigmoid()

    wdl_value_target = wdl_eval_target * (1.0 - wdl) + outcome * wdl
    
    return torch.abs(wdl_value_target  - wdl_eval_model).square().mean()
