import argparse
import halfkp
import math
import model as M
import numpy
import struct
import torch

VERSION = 0x00000003

  
def write_header(buf):
    buf.extend(struct.pack('<i', VERSION))


def write_feature_transformer(buf, model):
    layer = model.input
    bias = layer.bias.data
    buf.extend(bias.flatten().numpy().tobytes())
    weight = model.combine_feature_weights()
    buf.extend(weight.transpose(0, 1).flatten().numpy().tobytes())


def write_fc_layer(buf, layer):
    bias = layer.bias.data
    buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data
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
    write_fc_layer(buf, nnue.output)

    # Write converted model
    with open(target, 'wb') as f:
        f.write(buf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert models from .pt format to .nnue format.')
    parser.add_argument('source', help='Source file (.pt)')
    parser.add_argument('target', help='Target file (.nnue)')
    args = parser.parse_args()

#    print('Converting %s to %s' % (args.source, args.target))
    serialize(args.source, args.target)
