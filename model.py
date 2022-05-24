import chess
import halfkp
import torch
import struct
from torch import nn
import torch.nn.functional as F

# The version of the export format
EXPORT_FORMAT_VERSION = 0x00000004

# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32

class NNUE(nn.Module):
    def __init__(self, use_factorizer=False, feature_set=halfkp.Features()):
        super(NNUE, self).__init__()
        self.use_factorizer = use_factorizer
        self.feature_set = feature_set
        self.input = nn.Linear(feature_set.get_num_inputs(use_factorizer), L1)
        self.l1 = nn.Linear(2 * L1, L2)
        self.l2 = nn.Linear(L2, L3)
        self.output = nn.Linear(L3, 1)

        if self.use_factorizer:
            self.init_virtual_features()


    def forward(self, us, them, w_in, b_in):
        w = self.input(w_in)
        b = self.input(b_in)
        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        l0_ = torch.clamp(l0_, 0.0, 1.0)
        l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
        l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
        x = self.output(l2_)
        return x


    def init_virtual_features(self):
        weights = self.input.weight.clone()
        first, last = self.feature_set.get_virtual_features_indices()
        weights[:, first:last] = 0.0
        self.input.weight = nn.Parameter(weights)


    def combine_feature_weights(self):
        if not self.use_factorizer:
            return self.input.weight

        weights = self.input.weight.data.clone()
        combined_weight = weights.new_zeros((weights.shape[0], halfkp.NUM_REAL_FEATURES))

        for real_idx in range(halfkp.NUM_REAL_FEATURES):
            virtual_idx = self.feature_set.real_to_virtual_feature(real_idx)
            combined_weight[:, real_idx] = weights[:, real_idx] + weights[:, virtual_idx]

        return combined_weight


    def serialize_halfkx_layer(self, buf, layer):
        bias = layer.bias.data.cpu()
        buf.extend(bias.flatten().numpy().tobytes())
        weight = self.combine_feature_weights().cpu()
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
