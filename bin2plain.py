#!/usr/bin/env python3
import argparse
import nnue_bin_reader

def main(args):
    # Create a reader for the input file
    reader = nnue_bin_reader.NNUEBinReader(args.input)

    # Open output file
    writer = open(args.output, 'w')

    # Iterate over all samples
    nsamples = reader.get_num_samples()
    i = 0
    while i < nsamples:
        # Read sample
        board, score, move, ply, result = reader.get_sample()
        
        # Write sample
        fenstr = board.fen(en_passant='fen')
        movestr = move.uci()
        writer.write('fen ' + fenstr + '\n')
        writer.write('move ' + movestr + '\n')
        writer.write('score ' + str(score) + '\n')
        writer.write('ply ' + str(ply) + '\n')
        writer.write('result ' + str(result) + '\n')
        writer.write('e\n')

        i = i + 1

    # Close reader and writer
    writer.close()
    reader.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
            help='the name of the input file', required=True)
    parser.add_argument('-o', '--output', type=str,
            help='the name of the output file', required=True)
    args = parser.parse_args()

    main(args)
