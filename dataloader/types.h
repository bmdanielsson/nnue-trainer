/*
 * Copyright (C) 2021 Martin Danielsson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <stdbool.h>

#define NSQUARES  64
#define NPIECES   12
#define NSIDES    2
#define NO_SQUARE 64

/* The different files */
enum {
    FILE_A,
    FILE_B,
    FILE_C,
    FILE_D,
    FILE_E,
    FILE_F,
    FILE_G,
    FILE_H
};

/* The different ranks */
enum {
    RANK_1,
    RANK_2,
    RANK_3,
    RANK_4,
    RANK_5,
    RANK_6,
    RANK_7,
    RANK_8
};

/* The different squares */
enum {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8
};

/* The different sides */
enum {
    WHITE,
    BLACK
};

/* The different piece types */
enum {
    PAWN = 0,
    KNIGHT = 2,
    BISHOP = 4,
    ROOK = 6,
    QUEEN = 8,
    KING = 10,
    NO_PIECE_TYPE = 12
};

/* The different pieces */
enum {
    WHITE_PAWN,
    BLACK_PAWN,
    WHITE_KNIGHT,
    BLACK_KNIGHT,
    WHITE_BISHOP,
    BLACK_BISHOP,
    WHITE_ROOK,
    BLACK_ROOK,
    WHITE_QUEEN,
    BLACK_QUEEN,
    WHITE_KING,
    BLACK_KING,
    NO_PIECE
};

/* Flags indicating castling availability */
enum {
    WHITE_KINGSIDE = 1,
    WHITE_QUEENSIDE = 2,
    BLACK_KINGSIDE = 4,
    BLACK_QUEENSIDE = 8
};

#define SQUARE(f, r)   (((r)<<3)+(f))
#define PIECE(c, t)    ((t) + (c))
#define PIECE_TYPE(p)  ((p)&(~BLACK))
#define PIECE_COLOR(p) ((p)&BLACK)
#define RANKNR(s)      ((s)>>3)
#define FILENR(s)      ((s)&7)
#define MIRROR(s)      ((s)^56)

/* Flags for different move types */
enum {
    NORMAL = 0,
    CAPTURE = 1,
    PROMOTION = 2,
    EN_PASSANT = 4,
    KINGSIDE_CASTLE = 8,
    QUEENSIDE_CASTLE = 16,
    NULL_MOVE = 32
};

/*
 * Chess moves are represented using an unsigned 32-bit integer. The bits
 * are assigned as follows:
 *
 * bit 0-5: from square (0-63)
 * bit 6-11: to square (0-63)
 * bit 12-15: promoted piece (see pieces enum)
 * bit 16-21: move type flags (see move types enum)
 */
#define MOVE(f, t, p, l)        ((uint32_t)(((f)) | \
                                 ((t)<<6) | \
                                 ((p)<<12) | \
                                 ((l)<<16)))
#define NULLMOVE                MOVE(0, 0, NO_PIECE, NULL_MOVE)
#define FROM(m)                 ((int)((m)&0x0000003F))
#define TO(m)                   ((int)(((m)>>6)&0x0000003F))
#define PROMOTION(m)            ((int)(((m)>>12)&0x0000000F))
#define TYPE(m)                 ((int)(((m)>>16)&0x0000003F))
#define ISNORMAL(m)             (TYPE((m)) == 0)
#define ISCAPTURE(m)            ((TYPE((m))&CAPTURE) != 0)
#define ISPROMOTION(m)          ((TYPE((m))&PROMOTION) != 0)
#define ISENPASSANT(m)          ((TYPE((m))&EN_PASSANT) != 0)
#define ISKINGSIDECASTLE(m)     ((TYPE((m))&KINGSIDE_CASTLE) != 0)
#define ISQUEENSIDECASTLE(m)    ((TYPE((m))&QUEENSIDE_CASTLE) != 0)
#define ISNULLMOVE(m)           ((TYPE((m))&NULL_MOVE) != 0)
#define ISTACTICAL(m)           (ISCAPTURE(m)||ISENPASSANT(m)||ISPROMOTION(m))
#define NOMOVE                  0

/* Internal representation of a chess position */
struct position {
    uint8_t board[NSQUARES];
    uint8_t kings[NSIDES];
    uint8_t pieces[30];
    uint8_t npieces;
    uint8_t stm;
};

struct sparse_batch {
    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
};

#endif
