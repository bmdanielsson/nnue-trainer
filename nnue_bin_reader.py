import chess
import os
import math

import numpy as np

SFEN_BIN_SIZE = 40
HUFFMAN_TABLE = [
    (0b0000, 1, None),
    (0b0001, 4, chess.PAWN),
    (0b0011, 4, chess.KNIGHT),
    (0b0101, 4, chess.BISHOP),
    (0b0111, 4, chess.ROOK),
    (0b1001, 4, chess.QUEEN)
]

def read_bit(data, cursor):
    b = (data[cursor//8] >> (cursor&7))&1;
    cursor += 1
    return b, cursor


def read_bits(data, cursor, nbits):
    v = 0
    for k in range(nbits):
        b, cursor = read_bit(data, cursor)
        if b > 0:
            v |= (1 << k)
    return v, cursor


def read_piece(data, cursor):
    found = False
    code = 0
    nbits = 0
    piece_type = None
    while not found:
        b, cursor = read_bit(data, cursor)
        code |= (b << nbits)
        nbits += 1
        
        for k in range(6):
            if HUFFMAN_TABLE[k][0] == code and HUFFMAN_TABLE[k][1] == nbits:
                found = True
                piece_type = HUFFMAN_TABLE[k][2]
                break

    if not piece_type:
        return piece_type, cursor

    b, cursor = read_bit(data, cursor)
    if b == 0:
        color = chess.WHITE
    else:
        color = chess.BLACK

    return chess.Piece(piece_type, color), cursor


def read_position(data):
    cursor = 0
    board = chess.Board(fen=None)

    # Side to move
    b, cursor = read_bit(data, cursor)
    if b == 0:
        board.turn = chess.WHITE
    else:
        board.turn = chess.BLACK

    # King positions
    wksq, cursor = read_bits(data, cursor, 6)
    board.set_piece_at(wksq, chess.Piece(chess.KING, chess.WHITE))
    bksq, cursor = read_bits(data, cursor, 6)
    board.set_piece_at(bksq, chess.Piece(chess.KING, chess.BLACK))

    # Piece positions
    for r in range(8)[::-1]:
        for f in range(8):
            sq = chess.square(f, r)
            if sq == wksq or sq == bksq:
                continue

            piece, cursor = read_piece(data, cursor)
            if piece:
                board.set_piece_at(sq, piece)

    # Castling availability
    b, cursor = read_bit(data, cursor)
    if b == 1:
        board.castling_rights |= chess.BB_H1
    b, cursor = read_bit(data, cursor)
    if b == 1:
        board.castling_rights |= chess.BB_A1
    b, cursor = read_bit(data, cursor)
    if b == 1:
        board.castling_rights |= chess.BB_H8
    b, cursor = read_bit(data, cursor)
    if b == 1:
        board.castling_rights |= chess.BB_A8

    # En-passant square
    b, cursor = read_bit(data, cursor)
    if b == 1:
        board.ep_square, cursor = read_bits(data, cursor, 6)

    # 50-move counter, low-bits
    low50, cursor = read_bits(data, cursor, 6)

    # Fullmove counter
    low_full, cursor = read_bits(data, cursor, 8)
    high_full, cursor = read_bits(data, cursor, 8)
    board.fullmove_number = (high_full << 8) | low_full

    # 50-move counter, high-bits
    high50, cursor = read_bit(data, cursor)
    board.halfmove_clock = (high50 << 6) | low50

    return board


def get_sfen(fh):
    # Read the first 32 bytes which describe the position
    data = np.fromfile(fh, dtype='uint8', count=32)
    board = read_position(data)

    # Read the next two bytes which is score
    score = np.fromfile(fh, dtype='int16', count=1)

    # Read and skip the next four bytes which is the move and ply count
    np.fromfile(fh, dtype='uint8', count=4)
    
    # Read the next byte which is the result
    result = np.fromfile(fh, dtype='int8', count=1)

    # Read and skip one byte which is just padding
    np.fromfile(fh, dtype='uint8', count=1)

    return board, score, result


class NNUEBinReader:
    def __init__(self, path):
        self.path = path
        self.fh = open(path, 'rb')
        self.nsamples = os.path.getsize(path)//SFEN_BIN_SIZE


    def close(self):
        self.fh.close()


    def get_num_samples(self):
        return self.nsamples


    def get_sample(self):
        # Read the first 32 bytes which describe the position
        data = np.fromfile(self.fh, dtype='uint8', count=32)
        board = read_position(data)

        # Read the next two bytes which is score
        score = np.fromfile(self.fh, dtype='int16', count=1)

        # Read and skip the next four bytes which is the move and ply count
        np.fromfile(self.fh, dtype='uint8', count=4)
    
        # Read the next byte which is the result
        result = np.fromfile(self.fh, dtype='int8', count=1)

        # Read and skip one byte which is just padding
        np.fromfile(self.fh, dtype='uint8', count=1)

        return (board, score, result)
