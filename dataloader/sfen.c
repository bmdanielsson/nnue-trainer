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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "sfen.h" 
#include "board.h"

struct packed_sfen {
    uint8_t  position[32];
    int16_t  score;
    uint16_t move;
    uint16_t ply;
    int8_t   result;
    uint8_t  padding;
};

struct compressed_sfen {
    uint16_t marker;
    int16_t  score;
    uint16_t move;
    uint16_t ply;
    int8_t   result;
    uint8_t  padding;
};

/* Table containing Huffman encoding of each piece type */
static struct huffman_piece {
    int code;       /* Code for the piece */
    int nbits;      /* The number of bits in the code */
    int piece_type; /* The type of piece */
} huffman_table[] = {
    {0b0000, 1, NO_PIECE_TYPE}, /* No piece */
    {0b0001, 4, PAWN},          /* Pawn */
    {0b0011, 4, KNIGHT},        /* Knight */
    {0b0101, 4, BISHOP},        /* Bishop */
    {0b0111, 4, ROOK},          /* Rook */
    {0b1001, 4, QUEEN},         /* Queen */
};

static int read_bit(uint8_t *data, int *cursor)
{
    int b = (data[*cursor/8] >> (*cursor&7))&1;
    (*cursor)++;

    return b;
}

static int read_bits(uint8_t *data, int *cursor, int nbits)
{
    int k;
    int result = 0;

    for (k=0;k<nbits;k++) {
        result |= read_bit(data, cursor)?(1 << k):0;
    }

    return result;
}

static int read_piece(uint8_t *data, int *cursor)
{
    int  color = WHITE;
    int  code = 0;
    int  nbits = 0;
    bool found = false;
    int  k;

    while (!found) {
        code |= read_bit(data, cursor) << nbits;
        nbits++;

        for (k=0;k<6;k++) {
            if ((huffman_table[k].code == code) &&
                (huffman_table[k].nbits == nbits)) {
                found = true;
                break;
            }
        }
    }
        
    if (huffman_table[k].piece_type == NO_PIECE_TYPE) {
        return NO_PIECE;
    }

    color = read_bit(data, cursor);

    return PIECE(color, huffman_table[k].piece_type);
}

void read_position(uint8_t *data, int *cursor, struct position *pos)
{
    int sq;
    int rank;
    int file;
    int piece;
    int fifty;
    int fullmove;

    /* Initialize position */
    memset(pos, 0, sizeof(struct position));
    for (sq=0;sq<NSQUARES;sq++) {
        pos->board[sq] = NO_PIECE;
    }
    pos->ep_sq = NO_SQUARE;

    /* The side to move */
    pos->stm = read_bit(data, cursor);

    /* King positions */
    sq = read_bits(data, cursor, 6);
    pos->board[sq] = WHITE_KING;
    pos->pieces[pos->npieces++] = sq;
    sq = read_bits(data, cursor, 6);
    pos->board[sq] = BLACK_KING;
    pos->pieces[pos->npieces++] = sq;

    /* Piece positions */
    for (rank=RANK_8;rank>=RANK_1;rank--) {
        for (file=FILE_A;file<=FILE_H;file++) {
            sq = SQUARE(file, rank);
            if (pos->board[sq] != NO_PIECE) {
                continue;
            }
               
            piece = read_piece(data, cursor);
            if (piece != NO_PIECE) {
                pos->board[sq] = piece;
                pos->pieces[pos->npieces++] = sq;
            }
        }
    }

    /* Castling */
    if (read_bit(data, cursor) == 1) {
        pos->castle |= WHITE_KINGSIDE;
    }
    if (read_bit(data, cursor) == 1) {
        pos->castle |= WHITE_QUEENSIDE;
    }
    if (read_bit(data, cursor) == 1) {
        pos->castle |= BLACK_KINGSIDE;
    }
    if (read_bit(data, cursor) == 1) {
        pos->castle |= BLACK_QUEENSIDE;
    }

    /* En-passant square */
    if (read_bit(data, cursor) == 1) {
        pos->ep_sq = read_bits(data, cursor, 6);
    }

    /* 50-move counter, lower 6 bits */
    fifty = read_bits(data, cursor, 6);

    /* Fullmove counter */
    fullmove = read_bits(data, cursor, 8);
    fullmove |= (read_bits(data, cursor, 8) << 8);
    pos->fullmove = fullmove;

    /* 50-move counter, upper 1 bit */
    fifty |= (read_bit(data, cursor) << 6);
    pos->fifty = fifty;
}

static uint32_t parse_move(uint16_t packed_move, struct position *pos)
{
    int to = packed_move & 0x003F;
    int from = (packed_move >> 6) & 0x003F;
    int promotion = (packed_move >> 12) & 0x0003;
    int special = (packed_move >> 14) & 0x0003;
    int to_sq;
    int from_sq;
    int promotion_piece = NO_PIECE;
    int move_type = NORMAL;

    to_sq = to;
    from_sq = from;
    if (special == 1) {         /* Promotion */
        promotion_piece = promotion*2 + 2 + pos->stm;
        move_type = PROMOTION;
        if (pos->board[to_sq] != NO_PIECE) {
            move_type |= CAPTURE;
        }
    } else if (special == 2) {  /* En-passant */
        move_type = EN_PASSANT;
    } else if (special == 3) {  /* Castling */
        if (to < from) {
            move_type = QUEENSIDE_CASTLE;
        } else if (to > from) {
            move_type = KINGSIDE_CASTLE;
        }
    } else {
        if (pos->board[to_sq] != NO_PIECE) {
            move_type |= CAPTURE;
        }
    }

    return MOVE(from_sq, to_sq, promotion_piece, move_type);
}

void sfen_unpack_bin(uint8_t *data, struct sfen *sfen, struct position *pos)
{
    struct packed_sfen *packed = (struct packed_sfen*)data;
    int                cursor = 0;

    read_position(packed->position, &cursor, pos);

    sfen->pos = *pos;
    sfen->score = packed->score;
    sfen->move = parse_move(packed->move, &sfen->pos);
    sfen->ply = packed->ply;
    sfen->result = packed->result;
    assert(packed->padding == 0xFF);
}
