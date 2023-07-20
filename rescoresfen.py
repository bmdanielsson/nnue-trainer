#!/usr/bin/env python3
import sys
import random
import argparse
import random
import os
import shutil
import time
import subprocess
import nnue_bin_reader

from multiprocessing import Process, Lock, Value
from datetime import datetime

BATCH_SIZE = 10000
PROGRESS_INTERVAL = 10

def rescore_batch(outputfile, size, offset, args):
    # Setup command
    command = []

    command.append(args.engine)
    command.append('--rescore')

    command.append('--input')
    command.append(args.input)

    command.append('--output')
    command.append(outputfile)

    command.append('--depth')
    command.append(str(args.depth))

    command.append('--npositions')
    command.append(str(size))

    command.append('--offset')
    command.append(str(offset))

    # Execute command
    subprocess.call(command)


def update_work_status(finished, remaining_work, finished_work, position_lock):
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
                 position_lock, offset, samples_to_process, args):
    # Process all position assigned to this thread
    work_counter = 0
    work_done = 0
    while work_done < samples_to_process:
        size = BATCH_SIZE;
        if (work_done + size) > samples_to_process:
            size = samples_to_process - work_done
        rescore_batch(training_file, size, offset, args)
        work_done += size
        work_counter += size
        offset += size
        update_work_status(size, remaining_work, finished_work,
                            position_lock)

def main(args):
    before = datetime.now();

    # Find the starting offset
    if args.offset:
        start_offset = args.offset
    else:
        start_offset = 0

    # Find the number of samples to process
    reader = nnue_bin_reader.NNUEBinReader(args.input)
    nsamples = reader.get_num_samples()
    if args.npositions:
        total_to_process = args.npositions
    else:
        total_to_process = nsamples
    if total_to_process > nsamples:
        total_to_process = nsamples
    if (start_offset + total_to_process) > nsamples:
        total_to_process = nsamples - start_offset
    if total_to_process <= 0:
        exit(1)
    reader.close()
    samples_per_thread = total_to_process//args.nthreads

    # Initialize
    remaining_work = Value('i', total_to_process)
    finished_work = Value('i', 0)
    position_lock = Lock()

    # Start generating data with all threads
    training_files = []
    processes = []
    parts = os.path.splitext(args.output)
    for pid in range(0, args.nthreads):
        offset = start_offset + pid*samples_per_thread
        samples_to_process = samples_per_thread
        if pid == (args.nthreads-1):
            samples_to_process += (total_to_process%args.nthreads)
        training_file = parts[0] + '_' + str(pid) + parts[1]
        training_files.append(training_file)

        process_args = (pid, training_file, remaining_work, finished_work,
                        position_lock, offset, samples_to_process, args)
        processes.append(Process(target=process_func, args=process_args))
        processes[pid].start()

    # Handle progress output
    while True:
        time.sleep(PROGRESS_INTERVAL)

        position_lock.acquire()
        pos_rescored = finished_work.value
        position_lock.release()
        progress = (pos_rescored/total_to_process)*100
        print(f'\r{pos_rescored}/{total_to_process} ({int(progress)}%)', end='')

        if pos_rescored == total_to_process:
            break;
    print('\n');

    # Wait for all threads to exit
    for p in processes:
        p.join()

    # Merge data from all threads into one file
    with open(args.output, 'ab') as wfd:
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
    parser.add_argument('-e', '--engine', type=str,
            help='the engine to use', required=True)
    parser.add_argument('-d', '--depth', type=int, default=8,
            help='the depth to search each position to (deafult 8)')
    parser.add_argument('-t', '--nthreads', type=int, default='1',
            help='the number of threads to use (default 1)')
    parser.add_argument('-i', '--input', type=str,
            help='the name of the input file', required=True)
    parser.add_argument('-o', '--output', type=str,
            help='the name of the output file', required=True)
    parser.add_argument('-f', '--offset', type=int,
            help='the ofset of the first position to rescore (default 0=')
    parser.add_argument('-n', '--npositions', type=int,
            help='the number of positions to rescore (all)')

    args = parser.parse_args()

    print(f'Engine: {args.engine}')
    print(f'Depth: {args.depth}')
    print('')

    main(args)
