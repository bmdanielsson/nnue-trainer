import chess
import os
import math
import nnue_bin_writer

import numpy as np


class NNUEBinpackWriter(nnue_bin_writer.NNUEBinWriter):
    def __init__(self, path):
        nnue_bin_writer.NNUEBinWriter.__init__(self, path)
        self.board = None


    # position (256 bits) or 0x0000 (16 bits)
    # score (16 bits)
    # move (16 bits)
    # ply (16 bits)
    # result (8 bits)
    # padding (8 bits)
    def write_sample(self, position, score, move, ply, result, frc_pos):
        if self.board and self.board.fen() == position:
            pack = True
        else:
            pack = False
            self.board = chess.Board(fen=position, chess960=frc_pos)

        if pack:
            stm_result = result
            if self.board.turn == chess.BLACK:
                stm_result = -1*stm_result
            stm_score = score.pov(self.board.turn).score()

            np.int16(0x0000).tofile(self.fh)
            np.int16(stm_score).tofile(self.fh)
            move_data = self.encode_move(self.board, move)
            move_data.tofile(self.fh)
            np.uint16(ply).tofile(self.fh)
            np.int8(stm_result).tofile(self.fh)
            np.uint8(0xFF).tofile(self.fh)
        else:
            super().write_sample(position, score, move, ply, result, frc_pos)
        
        self.board.push(move)
