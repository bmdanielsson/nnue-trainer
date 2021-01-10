NUM_SQ = 64
NUM_PT = 10
NUM_PLANES = (NUM_SQ * NUM_PT + 1)

class Features:
  name = 'HalfKP'
  inputs = NUM_PLANES * NUM_SQ # 41024
