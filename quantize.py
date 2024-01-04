import argparse
import math
import model
import numpy
import struct
import torch

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

NNUE_FORMAT_VERSION = 0x00000008

def write_header(buf, version):
    buf.extend(struct.pack('<I', version))


def write_layer(buf, biases, weights):
    buf.extend(biases.numpy().tobytes())
    buf.extend(weights.numpy().tobytes())


def quant_input(biases, weights):
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


def extract_layer(file, ninputs, size):
    buf = numpy.fromfile(file, numpy.float32, size)
    biases = torch.from_numpy(buf.astype(numpy.float32))
    buf = numpy.fromfile(file, numpy.float32, size*ninputs)
    weights = torch.from_numpy(buf.astype(numpy.float32))
    return (biases, weights)


def quantization(source, target):
    print('Performing quantization ...')

    # Load model
    nnue = model.NNUE()
    nnue.load_state_dict(torch.load(source, map_location=torch.device('cpu')))
    nnue.eval()

    # Perform quantization
    input = quant_input(nnue.input.weight, nnue.input.bias)
    linear1 = quant_linear(nnue.l1.weight, nnue.l1.bias)
    linear2 = quant_linear(nnue.l2.weight, nnue.l2.bias)
    output = quant_output(nnue.output.weight, nnue.output.bias)

    # Write quantized layers
    outbuffer = bytearray()
    write_header(outbuffer, NNUE_FORMAT_VERSION)
    write_layer(outbuffer, input[0], input[1])
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
