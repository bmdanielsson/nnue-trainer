import chess
import torch
import struct
from torch import nn
import torch.nn.functional as F

# The version of the export format
EXPORT_FORMAT_VERSION = 0x00000008

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


    def serialize_halfkx_layer(self, buf, layer):
        bias = layer.bias.data.cpu()
        buf.extend(bias.flatten().numpy().tobytes())
        weight = self.input.weight.data.clone().cpu()
        buf.extend(weight.transpose(0, 1).flatten().numpy().tobytes())


    def serialize_linear_layer(self, buf, layer):
        bias = layer.bias.data.cpu()
        buf.extend(bias.flatten().numpy().tobytes())
        weight = layer.weight.data.cpu()
        buf.extend(weight.flatten().numpy().tobytes())


    def serialize(self, buf):
        # Write header
        buf.extend(struct.pack('<i', EXPORT_FORMAT_VERSION))

        # Write layers
        self.serialize_halfkx_layer(buf, self.input)
        self.serialize_linear_layer(buf, self.l1)
        self.serialize_linear_layer(buf, self.l2)
        self.serialize_linear_layer(buf, self.output)


def loss_function(lambda_, pred, batch):
    us, them, white, black, outcome, score = batch
    
    wdl_eval_model = (pred*600.0/361).sigmoid()
    wdl_eval_target = (score/410).sigmoid()

    wdl_value_target  = wdl_eval_target * lambda_ + outcome * (1.0 - lambda_)
    
    return torch.abs(wdl_value_target  - wdl_eval_model).square().mean()
