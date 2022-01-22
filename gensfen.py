#!/usr/bin/env python3
import sys
import random
import argparse
import random
import os
import shutil
import time
import chess
import chess.engine
import chess.syzygy
import chess.polyglot
import nnue_bin_writer
import nnue_binpack_writer

from multiprocessing import Process, Lock, Value
from datetime import datetime

BATCH_SIZE = 1000
PROGRESS_INTERVAL = 10
MAX_TIME = 30

RANDOM_PLIES = [10, 12, 14, 16]
MAX_PLY = 400
MIN_DRAW_PLY = 80
DRAW_SCORE = 10
DRAW_COUNT = 10
EVAL_LIMIT = 10000

def write_sfen_bin(writer, sfen, result):
    writer.write_sample(sfen['fen'], sfen['score'], sfen['move'], sfen['ply'],
                        result)


def write_sfen_plain(fh, sfen, result):
    stm_result = result
    if sfen['score'].turn == chess.BLACK:
        stm_result = -1*stm_result
    stm_score = sfen['score'].pov(sfen['score'].turn).score()

    fh.write('fen ' + sfen['fen'] + '\n')
    fh.write('move ' + sfen['move'].uci() + '\n')
    fh.write('score ' + str(stm_score) + '\n')
    fh.write('ply ' + str(sfen['ply']) + '\n')
    fh.write('result ' + str(stm_result) + '\n')
    fh.write('e\n')


def play_random_moves(board):
    nplies = random.choice(RANDOM_PLIES)
    for k in range(0, nplies):
        legal_moves = [move for move in board.legal_moves]
        if len(legal_moves) > 1:
            idx = random.randint(0, len(legal_moves)-1)
        else:
            idx = 0
        move = legal_moves[idx]
        board.push(move)
        if board.is_game_over(claim_draw=True):
            break


def setup_board(supports_frc, args):
    if args.frc_prob != 0.0 and random.random() < args.frc_prob:
        frc_board = chess.Board.from_chess960_pos(random.randint(0, 959))
        if not supports_frc:
            frc_board.set_castling_fen('-')
            board = chess.Board(fen=frc_board.fen())
        else:
            board = frc_board
    else:
        board = chess.Board()

    return board


def is_quiet(board, move):
    return (not move.promotion and
            not board.is_capture(move) and
            not board.is_en_passant(move) and
            not board.is_check() and
            not board.gives_check(move))


def play_game(writer, duplicates, hasher, pos_left, supports_frc, args):
    # Setup a new board
    board = setup_board(supports_frc, args)

    # Play a random opening
    play_random_moves(board)
    if board.is_game_over(claim_draw=True):
        return pos_left

    # Setup engine options
    options = {}
    options['Threads'] = 1
    options['OwnBook'] = False

    # Start engine playing white
    engine_white = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine_white.configure(options)

    # Start engine playing black
    engine_black = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine_black.configure(options)

    # Setup search limit
    limit = chess.engine.Limit(depth=args.depth, time=MAX_TIME)

    # Let the engine play against itself and a record all positions
    resign_count = 0
    draw_count = 0 
    count = 0
    positions = []
    result_val = 0
    while not board.is_game_over(claim_draw=True):
        # Search the position to the required depth
        if board.turn == chess.WHITE:
            result = engine_white.play(board, limit,
                                       info=chess.engine.Info.SCORE)
        else:
            result = engine_black.play(board, limit,
                                       info=chess.engine.Info.SCORE)

        # If no score was received then skip this move
        if 'score' not in result.info:
            board.push(result.move)
            continue

        # Skip non-quiet moves
        if not is_quiet(board, result.move):
            board.push(result.move)
            continue

        # Check eval limit
        if abs(result.info['score'].relative.score()) >= EVAL_LIMIT:
            if result.info['score'].white().score() > 0:
                result_val = 1
            else:
                result_val = -1
            break

        # Check for duplicates
        boardhash = hasher.hash_board(board)
        if boardhash in duplicates:
            board.push(result.move)
            continue
        else:
            duplicates.add(boardhash)

        # Extract and store information
        if board.turn == chess.WHITE:
            ply = board.fullmove_number*2
        else: 
            ply = board.fullmove_number*2 + 1
        sfen = {'fen':board.fen(en_passant='fen'), 'move':result.move,
                'score':result.info['score'], 'ply':ply}
        positions.append(sfen)

        # Check ply limit
        if ply > MAX_PLY:
            result_val = 0
            break

        # Check draw adjudication
        if ply > MIN_DRAW_PLY:
            if abs(result.info['score'].relative.score()) <= DRAW_SCORE:
                draw_count += 1;
            else:
                draw_count = 0
            if draw_count >= DRAW_COUNT:
                result_val = 0
                break;

        # Apply move
        board.push(result.move)

    engine_white.quit()
    engine_black.quit()

    # Convert result to sfen representation
    if board.is_game_over(claim_draw=True):
        result_str = board.result(claim_draw=True)
        if result_str == '1-0':
            result_val = 1
        elif result_str == '0-1':
            result_val = -1
        elif result_str == '1/2-1/2':
            result_val = 0

    # Write positions to file
    for sfen in positions:
        if args.format == 'plain':
            write_sfen_plain(writer, sfen, result_val)
        else:
            write_sfen_bin(writer, sfen, result_val)
        pos_left = pos_left - 1
        if pos_left == 0:
            break
    writer.flush()

    return pos_left


def request_work(finished, remaining_work, finished_work, position_lock):
    position_lock.acquire()
    finished_work.value = finished_work.value + finished
    if remaining_work.value == 0:
        npos = 0
    elif remaining_work.value < BATCH_SIZE:
        npos = remaining_work.value
    else:
        npos = BATCH_SIZE
    remaining_work.value = remaining_work.value - npos
    position_lock.release()
    return npos


def process_func(pid, training_file, remaining_work, finished_work,
                 position_lock, supports_frc, args):
    # Initialize variables for keeping track of duplicates
    duplicates = set()
    hasher = chess.polyglot.ZobristHasher(chess.polyglot.POLYGLOT_RANDOM_ARRAY)
        
    # Set seed for random number generation
    if (args.seed):
        random.seed(a=args.seed+pid*10)
    else:
        random.seed()

    # Open output file
    if args.format == 'plain':
        writer = open(training_file, 'w')
    elif args.format == 'bin':
        writer = nnue_bin_writer.NNUEBinWriter(training_file)
    else:
        writer = nnue_binpack_writer.NNUEBinpackWriter(training_file)

    # Keep generating positions until the requested number is reached
    work_todo = 0
    while True:
        work_todo = request_work(work_todo, remaining_work, finished_work,
                                position_lock)
        if work_todo == 0:
            break
        pos_left = work_todo
        while pos_left > 0:
            pos_left = play_game(writer, duplicates, hasher, pos_left,
                                supports_frc, args)

    writer.close()


def main(args):
    before = datetime.now();

    # Initialize
    remaining_work = Value('i', args.npositions)
    finished_work = Value('i', 0)
    position_lock = Lock()

    # Check if the engine supports FRC
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    supports_frc = 'UCI_Chess960' in engine.options
    engine.quit()

    # Start generating data with all threads
    training_files = []
    processes = []
    parts = os.path.splitext(args.output)
    for pid in range(0, args.nthreads):
        training_file = parts[0] + '_' + str(pid) + parts[1]
        training_files.append(training_file)

        process_args = (pid, training_file, remaining_work, finished_work,
                        position_lock, supports_frc, args)
        processes.append(Process(target=process_func, args=process_args))
        processes[pid].start()

    # Handle progress output
    while True:
        time.sleep(PROGRESS_INTERVAL)

        position_lock.acquire()
        pos_generated = finished_work.value
        position_lock.release()
        print(f'\r{pos_generated}/{args.npositions}', end='')

        if pos_generated == args.npositions:
            break;
    print('\n');

    # Wait for all threads to exit
    for p in processes:
        p.join()

    # Merge data from all threads into one file
    if args.format == 'plain':
        with open(args.output, 'w') as wfd:
            for f in training_files:
                with open(f, 'r') as fd:
                    shutil.copyfileobj(fd, wfd)
    else:
        with open(args.output, 'wb') as wfd:
            for f in training_files:
                with open(f, 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
    for f in training_files:
        os.remove(f)

    after = datetime.now();
    print(f'Time: {after-before}')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--depth', type=int, default=8,
            help='the depth to search each position to (deafult 8)')
    parser.add_argument('-e', '--engine', type=str, required=True,
            help='the path to the engine')
    parser.add_argument('-t', '--nthreads', type=int, default='1',
            help='the number of threads to use (default 1)')
    parser.add_argument('-n', '--npositions', type=int, default='100000000',
            help='the number of positions to generate (default 100000000)')
    parser.add_argument('-o', '--output', type=str,
            help='the name of the output file', required=True)
    parser.add_argument('--format', choices=['plain', 'bin', 'binpack'],
            default='bin', help='the output format (default bin)')
    parser.add_argument('--seed', type=int,
            help='seed to use for random number generator')
    parser.add_argument('--frc-prob', type=float, default=0.0,
            help="Probability of using a FRC starting position (default 0.0)")

    args = parser.parse_args()

    print(f'Engine: {args.engine}')
    print(f'Output format: {args.format}')
    print(f'Number of positions: {args.npositions}')
    print(f'Depth: {args.depth}')
    print(f'FRC probabliity: {args.frc_prob}')
    print('')

    main(args)
