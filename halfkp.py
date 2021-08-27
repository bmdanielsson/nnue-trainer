NUM_SQ = 64
NUM_PT = 10
NUM_PLANES = NUM_SQ*NUM_PT
NUM_REAL_FEATURES = NUM_PLANES*NUM_SQ
NUM_VIRTUAL_FEATURES = NUM_PT*NUM_SQ

class Features:
    name = 'HalfKP'


    def get_num_inputs(self, use_factorizer):
        if use_factorizer:
            ninputs = NUM_REAL_FEATURES + NUM_VIRTUAL_FEATURES
        else:
            ninputs = NUM_REAL_FEATURES
        return ninputs


    def get_virtual_features_indices(self):
        return NUM_REAL_FEATURES, NUM_REAL_FEATURES + NUM_VIRTUAL_FEATURES


    def real_to_virtual_feature(self, real_idx):
        piece_idx = real_idx%NUM_PLANES
        return NUM_REAL_FEATURES + piece_idx
