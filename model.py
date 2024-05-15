import torch
import struct

from torch import nn


# Number of inputs
NUM_SQ = 64
NUM_PT = 12
NUM_INPUTS = NUM_SQ*NUM_PT

L1 = 1024

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.input = nn.Linear(NUM_INPUTS, L1)
        self.output = nn.Linear(2*L1, 1)


    def forward(self, us, them, w_in, b_in):
        w = self.input(w_in)
        b = self.input(b_in)
        l0_ = (us*torch.cat([w, b], dim=1)) + (them*torch.cat([b, w], dim=1))
        l0_ = torch.clamp(l0_, 0.0, 1.0)
        x = self.output(l0_)
        return x


def loss_function(wdl, pred, batch):
    us, them, white, black, outcome, score = batch
    
    wdl_eval_model = (pred*600.0/361).sigmoid()
    wdl_eval_target = (score/410).sigmoid()

    wdl_value_target = wdl_eval_target * (1.0 - wdl) + outcome * wdl
    
    return torch.abs(wdl_value_target  - wdl_eval_model).square().mean()
