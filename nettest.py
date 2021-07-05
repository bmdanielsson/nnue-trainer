import chess
import os
import math
import argparse
import numpy as np
import os.path
import subprocess
import shutil

import serialize


def add_engine_options(command, engine, net):
    command.append('-engine')
    command.append('cmd=' + engine)
    command.append('name=' + net)
    command.append('option.EvalFile=' + net)
    command.append('option.UseNNUE=true')


def run_match(args, test_net):
    command = []

    command.append('c-chess-cli')

    test_dir_abs_path = os.path.abspath(args.test_engine)
    test_dir_abs_path = os.path.dirname(test_dir_abs_path)

    test = os.path.basename(test_net)
    test_abs_path = os.path.join(test_dir_abs_path, test)

    shutil.copy(test_net, test_abs_path)

    add_engine_options(command, args.base_engine, args.base_net)
    add_engine_options(command, args.test_engine, test)

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

    os.remove(test_abs_path)


def main(args):
    # Find all snapshot files in the net folder
    snapshot_files = [f for f in os.listdir(args.net_dir)
                       if (os.path.isfile(os.path.join(args.net_dir, f)) and
                           os.path.splitext(f)[1] == '.pt')]

    # Create temp folder if it doesn't exist
    if not os.path.exists(args.temp_dir):
        os.mkdir(args.temp_dir)

    # Convert each snapshot to nnue format and store in temp folder
    nnue_files = []    
    for snapshot in snapshot_files:
        in_path = os.path.join(args.net_dir, snapshot)
        out_path = os.path.join(args.temp_dir,
                                os.path.splitext(snapshot)[0]+'.nnue')
        out_path = out_path.replace('=', '_') 
        serialize.serialize(in_path, out_path)
        nnue_files.append(out_path)

    for nnue in nnue_files:
        run_match(args, nnue)

    # Remove temp directory
    shutil.rmtree(args.temp_dir)
        

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-engine', help='the path to the base engine',
                        type=str, required=True)
    parser.add_argument('--base-net', help='the base net to compare against',
                        type=str, required=True)
    parser.add_argument('--test-engine', help='the path to the test engine',
                        type=str, required=True)
    parser.add_argument('--net-dir', help='folder containing nets to test',
                        type=str, required=True)
    parser.add_argument('-t', '--temp-dir',
                    help='folder to store temporary files in (default temp)',
                    type=str, default='temp')
    parser.add_argument('-c', '--concurrency', type=int, default='1',
                    help='the number of games to run in parallell (default 1)')
    parser.add_argument('-g', '--games', type=int, default='200',
            help='the number of games to to play in each match (default 200)')
    parser.add_argument('-o', '--output',
                help='file to store all playes games in (deafult games.pgn)',
                type=str, default='games.pgn')

    args = parser.parse_args()

    main(args)
