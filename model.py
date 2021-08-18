import chess
import halfkp
import torch
from torch import nn
import torch.nn.functional as F

# 3 layer fully connected network
L1 = 128
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


def loss_function(lambda_, pred, batch):
    us, them, white, black, outcome, score = batch

    q = pred
    t = outcome
    p = (score / 600.0).sigmoid()
    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    result  = lambda_ * teacher_loss    + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    return loss
