import argparse
import halfkp
import math
import model
import numpy
import struct
import torch
import halfkp

NNUE2SCORE = 600.0
MAX_QUANTIZED_ACTIVATION = 127.0
WEIGHT_SCALE_BITS = 6
OUTPUT_SCALE = 16.0

HALFKX_WEIGHT_SCALE = MAX_QUANTIZED_ACTIVATION
HALFKX_BIAS_SCALE = MAX_QUANTIZED_ACTIVATION
HIDDEN_WEIGHT_SCALE = (1<<WEIGHT_SCALE_BITS)
HIDDEN_BIAS_SCALE = (1<<WEIGHT_SCALE_BITS)*MAX_QUANTIZED_ACTIVATION
OUTPUT_WEIGHT_SCALE = (OUTPUT_SCALE*NNUE2SCORE/MAX_QUANTIZED_ACTIVATION)
OUTPUT_BIAS_SCALE = OUTPUT_SCALE*NNUE2SCORE
MAX_HIDDEN_WEIGHT = MAX_QUANTIZED_ACTIVATION/HIDDEN_WEIGHT_SCALE
MAX_OUTPUT_WEIGHT = MAX_QUANTIZED_ACTIVATION/OUTPUT_WEIGHT_SCALE


def write_header(buf, version):
    buf.extend(struct.pack('<I', version))


def write_layer(buf, biases, weights):
    buf.extend(biases.numpy().tobytes())
    buf.extend(weights.numpy().tobytes())


def quant_halfkx(biases, weights):
    biases = biases.mul(HALFKX_BIAS_SCALE).round().to(torch.int16)
    weights = weights.mul(HALFKX_WEIGHT_SCALE).round().to(torch.int16)
    return (biases, weights)


def quant_linear(biases, weights):
    biases = biases.mul(HIDDEN_BIAS_SCALE).round().to(torch.int32)
    weights = weights.clamp(-MAX_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT).mul(HIDDEN_WEIGHT_SCALE).round().to(torch.int8)
    return (biases, weights)


def quant_output(biases, weights):
    biases = biases.mul(OUTPUT_BIAS_SCALE).round().to(torch.int32)
    weights = weights.clamp(-MAX_OUTPUT_WEIGHT, MAX_OUTPUT_WEIGHT).mul(OUTPUT_WEIGHT_SCALE).round().to(torch.int8)
    return (biases, weights)


def read_version(file):
    version = struct.unpack('<I', file.read(4))[0]
    if version != model.EXPORT_FORMAT_VERSION:
        raise Exception('Model format mismatch')
    return version
    

def read_layer(file, ninputs, size):
    buf = numpy.fromfile(file, numpy.float32, size)
    biases = torch.from_numpy(buf.astype(numpy.float32))
    buf = numpy.fromfile(file, numpy.float32, size*ninputs)
    weights = torch.from_numpy(buf.astype(numpy.float32))
    return (biases, weights)


def quantization(source, target):
    print('Performing quantization ...')

    # Read all layers
    with open(source, 'rb') as f:
        version = read_version(f)
        halfkx = read_layer(f, halfkp.NUM_FEATURES, model.L1)
        linear1 = read_layer(f, model.L1*2, model.L2)
        linear2 = read_layer(f, model.L2, model.L3)
        output = read_layer(f, model.L3, 1)

    # Perform quantization
    halfkx = quant_halfkx(halfkx[0], halfkx[1])
    linear1 = quant_linear(linear1[0], linear1[1])
    linear2 = quant_linear(linear2[0], linear2[1])
    output = quant_output(output[0], output[1])

    # Write quantized layers
    outbuffer = bytearray()
    write_header(outbuffer, version)
    write_layer(outbuffer, halfkx[0], halfkx[1])
    write_layer(outbuffer, linear1[0], linear1[1])
    write_layer(outbuffer, linear2[0], linear2[1])
    write_layer(outbuffer, output[0], output[1])
    with open(target, 'wb') as f:
        f.write(outbuffer)
   
    print('Quantization done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform quantization.')
    parser.add_argument('source', help='Source file')
    parser.add_argument('target', help='Destination file')
    args = parser.parse_args()

    quantization(args.source, args.target)
