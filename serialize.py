import argparse
import halfkp
import math
import model as M
import numpy
import struct
import torch

VERSION = 0x00000001

  
def write_header(buf):
    buf.extend(struct.pack('<i', VERSION))


def write_feature_transformer(buf, model):
    # int16 bias = round(x * 127)
    # int16 weight = round(x * 127)
    layer = model.input
    bias = layer.bias.data
    bias = bias.mul(127).round().to(torch.int16)
    buf.extend(bias.flatten().numpy().tobytes())

    weight = model.combine_feature_weights()
    weight = weight.mul(127).round().to(torch.int16)
    buf.extend(weight.transpose(0, 1).flatten().numpy().tobytes())


def write_fc_layer(buf, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
        kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
        kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
    kMaxWeight = 127.0 / kWeightScale # roughly 2.0

    # int32 bias = round(x * kBiasScale)
    # int8 weight = round(x * kWeightScale)
    bias = layer.bias.data
    bias = bias.mul(kBiasScale).round().to(torch.int32)
    buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data
    weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
    # Stored as [outputs][inputs], so we can flatten
    buf.extend(weight.flatten().numpy().tobytes())


def serialize(source, target):
    # Load model in .pt format
    try:
        nnue = M.NNUE(feature_set=halfkp.Features())
        nnue.load_state_dict(torch.load(source,
                             map_location=torch.device('cpu')))
    except RuntimeError:
        nnue = M.NNUE(use_factorizer=True, feature_set=halfkp.Features())
        nnue.load_state_dict(torch.load(source,
                             map_location=torch.device('cpu')))
    nnue.eval()

    # Convert model to .nnue format
    buf = bytearray()
    write_header(buf)
    write_feature_transformer(buf, nnue)
    write_fc_layer(buf, nnue.l1)
    write_fc_layer(buf, nnue.l2)
    write_fc_layer(buf, nnue.output, is_output=True)

    # Write converted model
    with open(target, 'wb') as f:
        f.write(buf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert models from .pt format to .nnue format.')
    parser.add_argument('source', help='Source file (.pt)')
    parser.add_argument('target', help='Target file (.nnue)')
    args = parser.parse_args()

    print('Converting %s to %s' % (args.source, args.target))
    serialize(args.source, args.target)
