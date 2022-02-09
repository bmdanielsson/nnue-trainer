import chess
import os
import math

import numpy as np

HUFFMAN_TABLE = [np.uint8(0),    # No piece (1 bit)
                 np.uint8(1),    # Pawn (4 bits)
                 np.uint8(3),    # Knight (4 bits)
                 np.uint8(5),    # Bishop (4 bits)
                 np.uint8(7),    # Rook (4 bits)
                 np.uint8(9)]    # Queen (4 bits)


# Encode a single-bit value according to Stockfish bitstream format
def encode_bit(data, pos, value):
    if np.uint16(value) != np.uint16(0):
        data[int(pos/8)] = data[int(pos/8)] | (1 << (pos & np.uint16(7)))

    return pos + 1


# Encode a multi-bit value according to Stockfish bitstream format
def encode_bits(data, pos, value, nbits):
    for i in range(0, nbits):
        pos = encode_bit(data, pos, np.uint16(value & np.uint16(1 << i)))

    return pos


def encode_piece_at(data, pos, board, sq):
    piece_type = board.piece_type_at(sq)
    piece_color = board.color_at(sq)

    if piece_type == None:
        pos = encode_bits(data, pos, np.uint16(HUFFMAN_TABLE[0]), 1)
        return pos

    pos = encode_bits(data, pos, np.uint16(HUFFMAN_TABLE[piece_type]), 4)
    pos = encode_bit(data, pos, np.uint16(not piece_color))

    return pos


class NNUEBinWriter:
    def __init__(self, path):
        self.path = path
        self.fh = open(path, 'wb')


    def flush(self):
        self.fh.flush()


    def close(self):
        self.fh.close()


    # Side to move (White = 0, Black = 1) (1bit)
    # White King Position (6 bits)
    # Black King Position (6 bits)
    # Huffman Encoding of the board
    # Castling availability (1 bit x 4)
    # En passant square (1 or 1 + 6 bits)
    # 50-move counter low bits (6 bits)
    # Full move counter (16 bits)
    # 50-move counter high bit (1 bit)
    def encode_position(self, board):
        data = np.zeros(32, dtype='uint8')
        pos = np.uint16(0)

        # Encode side to move
        pos = encode_bit(data, pos, np.uint16(not board.turn))

        # Encode king positions
        pos = encode_bits(data, pos, np.uint16(board.king(chess.WHITE)), 6)
        pos = encode_bits(data, pos, np.uint16(board.king(chess.BLACK)), 6)

        # Encode piece positions
        for r in reversed(range(0, 8)):
            for f in range(0, 8):
                sq = r*8 + f
                pc = board.piece_at(sq)
                if pc:
                    if (pc.piece_type == chess.KING or
                        pc.piece_type == chess.KING):
                        continue
                pos = encode_piece_at(data, pos, board, sq)

        # Encode castling availability
        if board.chess960:
            pos = encode_bits(data, pos, np.uint16(0), 4)
        else:
            pos = encode_bit(data, pos,
                    np.uint16(board.has_kingside_castling_rights(chess.WHITE)))
            pos = encode_bit(data, pos,
                    np.uint16(board.has_queenside_castling_rights(chess.WHITE)))
            pos = encode_bit(data, pos,
                    np.uint16(board.has_kingside_castling_rights(chess.BLACK)))
            pos = encode_bit(data, pos,
                    np.uint16(board.has_queenside_castling_rights(chess.BLACK)))

        # Encode en-passant square
        if not board.ep_square:
            pos = encode_bit(data, pos, np.uint16(0))
        else:
            pos = encode_bit(data, pos, np.uint16(1))
            pos = encode_bits(data, pos, np.uint16(board.ep_square), 6)

        # Encode 50-move counter. To keep compatibility with Stockfish trainer
        # only 6 bits are stored now. The last bit is stored at the end.
        pos = encode_bits(data, pos, np.uint16(board.halfmove_clock), 6)

        # Encode move counter
        pos = encode_bits(data, pos, np.uint16(board.fullmove_number), 8)
        pos = encode_bits(data, pos, np.uint16(board.fullmove_number>>8), 8)

        # Upper bit of 50-move counter
        high_bit = (board.halfmove_clock >> 6) & 1
        pos = encode_bit(data, pos, np.uint16(high_bit))

        return data


    # bit  0- 5: destination square (from 0 to 63)
    # bit  6-11: origin square (from 0 to 63)
    # bit 12-13: promotion piece type - 2 (from KNIGHT-2 to QUEEN-2)
    # bit 14-15: special move flag: promotion (1), en passant (2), castling (3)
    def encode_move(self, board, move):
        data = np.uint16(0)

        to_sq = move.to_square
        from_sq = move.from_square
        if not board.chess960:
            if board.turn == chess.WHITE and board.is_kingside_castling(move):
                to_sq = 7
            elif board.turn == chess.WHITE and board.is_queenside_castling(move):
                to_sq = 0
            elif board.turn == chess.BLACK and board.is_kingside_castling(move):
                to_sq = 63
            elif board.turn == chess.BLACK and board.is_queenside_castling(move):
                to_sq = 56

        data = data | np.uint16(to_sq)
        data = data | np.uint16(from_sq << 6)
        if move.promotion:
            data = data | np.uint16((move.promotion-2) << 12)
            data = data | np.uint16(1 << 14)
        if board.is_en_passant(move):
            data = data | np.uint16(2 << 14)
        elif board.is_castling(move):
            data = data | np.uint16(3 << 14)

        return data


    # position (256 bits)
    # score (16 bits)
    # move (16 bits)
    # ply (16 bits)
    # result (8 bits)
    # padding (8 bits)
    def write_sample(self, position, score, move, ply, result, frc_pos):
        board = chess.Board(fen=position, chess960=frc_pos)

        stm_result = result
        if board.turn == chess.BLACK:
            stm_result = -1*stm_result
        stm_score = score.pov(board.turn).score()

        pos_data = self.encode_position(board)
        pos_data.tofile(self.fh)
        np.int16(stm_score).tofile(self.fh)
        move_data = self.encode_move(board, move)
        move_data.tofile(self.fh)
        np.uint16(ply).tofile(self.fh)
        np.int8(stm_result).tofile(self.fh)
        np.uint8(0xFF).tofile(self.fh)
