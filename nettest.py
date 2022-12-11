import chess
import os
import math
import argparse
import numpy as np
import os.path
import subprocess
import shutil

import quantize

BASE_PATH='base'
TEST_PATH='test'
ENGINE_NAME='marvin'

def add_engine_options(command, engine, net):
    command.append('-engine')
    command.append('cmd=' + engine)
    if net:
        name = os.path.basename(net)
        name = os.path.splitext(name)[0]
        name = os.path.splitext(name)[0]
        command.append('name=' + name)
        command.append('option.EvalFile=' + net)
    command.append('option.UseNNUE=true')


def run_match(args, test_net):
    net_path = os.path.abspath(test_net)
    quant_net_path = net_path + ".nnue"
    quantize.quantization(net_path, quant_net_path)

    command = []

    command.append('c-chess-cli')

    add_engine_options(command, BASE_PATH+'/'+ENGINE_NAME, None)
    add_engine_options(command, TEST_PATH+'/'+ENGINE_NAME, quant_net_path)

    command.append('-each')
    command.append('tc=10+0.1')
    command.append('-concurrency')
    command.append(str(args.concurrency))
    command.append('-repeat')
    command.append('-resign')
    command.append('count=5')
    command.append('score=900')
    command.append('-draw')
    command.append('number=50')
    command.append('count=5')
    command.append('score=20')
    command.append('-gauntlet')
    command.append('-rounds')
    command.append(str(args.games))
    command.append('-openings')
    command.append('file=2moves_v2.epd')
    command.append('order=random')
    command.append('-pgn')
    command.append(args.output)

    subprocess.call(command)

    os.remove(quant_net_path)


def main(args):
    # Find all .nnue files in the net folder
    nnue_files = [f for f in os.listdir(args.net_dir)
                       if (os.path.isfile(os.path.join(args.net_dir, f)) and
                           os.path.splitext(f)[1] == '.bin')]

    # Run a match with each net
    for nnue in nnue_files:
        path = os.path.join(args.net_dir, nnue)
        run_match(args, path)


if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net-dir', type=str, required=True,
                    help='folder containing nets to test')
    parser.add_argument('-c', '--concurrency', type=int, default='1',
                    help='the number of games to run in parallell (default 1)')
    parser.add_argument('-g', '--games', type=int, default='200',
            help='the number of games to to play in each match (default 200)')
    parser.add_argument('-o', '--output',
                help='file to store all playes games in (deafult games.pgn)',
                type=str, default='games.pgn')

    args = parser.parse_args()

    main(args)
