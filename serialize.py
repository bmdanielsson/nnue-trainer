import argparse
import halfkp
import math
import model as M
import numpy
import struct
import torch
from functools import reduce
import operator

VERSION = 0x00000001

def write_int32(buf, v):
  buf.extend(struct.pack('<i', v))
  
  
def write_header(buf, model):
  write_int32(buf, VERSION) # version


def write_feature_transformer(buf, model):
  # int16 bias = round(x * 127)
  # int16 weight = round(x * 127)
  layer = model.input
  bias = layer.bias.data
  bias = bias.mul(127).round().to(torch.int16)
  buf.extend(bias.flatten().numpy().tobytes())

  weight = layer.weight.data
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


def read_int32(f, expected=None):
  v = struct.unpack("<i", f.read(4))[0]
  if expected is not None and v != expected:
    raise Exception("Expected: %x, got %x" % (expected, v))
  return v


def read_header(f):
  read_int32(f, VERSION) # version


def create_tensor(f, dtype, shape):
  d = numpy.fromfile(f, dtype, reduce(operator.mul, shape, 1))
  d = torch.from_numpy(d.astype(numpy.float32))
  d = d.reshape(shape)
  return d


def read_feature_transformer(f, layer):
  layer.bias.data = create_tensor(f, numpy.int16, layer.bias.shape).divide(127.0)
  # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
  weights = create_tensor(f, numpy.int16, layer.weight.shape[::-1])
  layer.weight.data = weights.divide(127.0).transpose(0, 1)


def read_fc_layer(f, layer, is_output=False):
  # FC layers are stored as int8 weights, and int32 biases
  kWeightScaleBits = 6
  kActivationScale = 127.0
  if not is_output:
    kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
  else:
    kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
  kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers

  layer.bias.data = create_tensor(f, numpy.int32, layer.bias.shape).divide(kBiasScale)
  layer.weight.data = create_tensor(f, numpy.int8, layer.weight.shape).divide(kWeightScale)


def main(args):
  print('Converting %s to %s' % (args.source, args.target))
  
  # Set feature set
  features = halfkp
  features_name = features.Features.name
  
  # Convert model
  if args.source.endswith('.pt'):
    if not args.target.endswith('.nnue'):
      raise Exception('Target file must end with .nnue')
    nnue = M.NNUE(feature_set=features.Features())
    nnue.load_state_dict(torch.load(args.source))
    nnue.eval()
    
    buf = bytearray()
    write_header(buf, nnue)
    write_feature_transformer(buf, nnue)
    write_fc_layer(buf, nnue.l1)
    write_fc_layer(buf, nnue.l2)
    write_fc_layer(buf, nnue.output, is_output=True)

    with open(args.target, 'wb') as f:
      f.write(buf)
  elif args.source.endswith('.nnue'):
    if not args.target.endswith('.pt'):
      raise Exception('Target file must end with .pt')
    with open(args.source, 'rb') as f:
      nnue = M.NNUE(feature_set=features.Features())
      read_header(f)
      read_feature_transformer(f, nnue.input)
      read_fc_layer(f, nnue.l1)
      read_fc_layer(f, nnue.l2)
      read_fc_layer(f, nnue.output, is_output=True)
      
      torch.save(nnue.state_dict(), args.target)
  else:
    raise Exception('Invalid filetypes: ' + str(args))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert models between .pt and .nnue formats.')
  parser.add_argument('source', help='Source file (.pt or .nnue)')
  parser.add_argument('target', help='Target file (.pt or .nnue)')
  args = parser.parse_args()
  
  main(args)
