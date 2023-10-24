NUM_SQ = 64
NUM_PT = 10
NUM_PLANES = NUM_SQ*NUM_PT
NUM_FEATURES = NUM_PLANES*NUM_SQ

class Features:
    name = 'HalfKP'


    def get_num_inputs(self):
        return NUM_FEATURES
