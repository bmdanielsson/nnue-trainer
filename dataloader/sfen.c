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

struct packed_sfen {
    uint8_t  position[32];
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

    /* Initialize position */
    memset(pos, 0, sizeof(struct position));
    for (sq=0;sq<NSQUARES;sq++) {
        pos->board[sq] = NO_PIECE;
    }

    /* The side to move */
    pos->stm = read_bit(data, cursor);

    /* King positions */
    sq = read_bits(data, cursor, 6);
    pos->board[sq] = WHITE_KING;
    pos->kings[WHITE] = sq;
    sq = read_bits(data, cursor, 6);
    pos->board[sq] = BLACK_KING;
    pos->kings[BLACK] = sq;

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

    /*
     * Remaining parts are skipped since
     * they are not used for training.
     */
}

void sfen_unpack_bin(uint8_t *data, struct sfen *sfen)
{
    struct packed_sfen *packed = (struct packed_sfen*)data;
    int                cursor = 0;

    read_position(packed->position, &cursor, &sfen->pos);
    sfen->score = packed->score;
    sfen->result = packed->result;

    /*
     * Remaining parts are ignored since
     * they are not used for training.
     */
}
