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
#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "board.h"

static char piece2char[NPIECES+1] = {
    'P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q', 'K', 'k', '.'
};

/* Destination squares for the king when castling */
static int kingside_castle_to[NSIDES] = {G1, G8};
static int queenside_castle_to[NSIDES] = {C1, C8};

static int to_square(uint32_t move)
{
    int to;

    to = TO(move);
    if (ISKINGSIDECASTLE(move)) {
        to = kingside_castle_to[to >= A8];
    } else if (ISQUEENSIDECASTLE(move)) {
        to = queenside_castle_to[to >= A8];
    }

    return to;
}

void pos2str(struct position *pos, char *str)
{
    char *iter;
    int  empty_count;
    int  rank;
    int  file;
    int  sq;

    /* Clear the string */
    memset(str, 0, FEN_MAX_LENGTH);

    /* Piece placement */
    empty_count = 0;
    iter = str;
    for (rank=RANK_8;rank>=RANK_1;rank--) {
        for (file=FILE_A;file<=FILE_H;file++) {
            sq = SQUARE(file, rank);
            if (pos->board[sq] != NO_PIECE) {
                if (empty_count > 0) {
                    *(iter++) = '0' + empty_count;
                    empty_count = 0;
                }
                *(iter++) = piece2char[pos->board[sq]];
            } else {
                empty_count++;
            }
        }
        if (empty_count != 0) {
            *(iter++) = '0' + empty_count;
            empty_count = 0;
        }
        if (rank > 0) {
            *(iter++) = '/';
        }
    }
    *(iter++) = ' ';

    /* Active color */
    if (pos->stm == WHITE) {
        *(iter++) = 'w';
    } else {
        *(iter++) = 'b';
    }
    *(iter++) = ' ';

    /* Castling avliability */
    if (pos->castle == 0) {
        *(iter++) = '-';
    } else {
        if (pos->castle&WHITE_KINGSIDE) {
            *(iter++) = 'K';
        }
        if (pos->castle&WHITE_QUEENSIDE) {
            *(iter++) = 'Q';
        }
        if (pos->castle&BLACK_KINGSIDE) {
            *(iter++) = 'k';
        }
        if (pos->castle&BLACK_QUEENSIDE) {
            *(iter++) = 'q';
        }
    }
    *(iter++) = ' ';

    /* En passant target square */
    if (pos->ep_sq == NO_SQUARE) {
        *(iter++) = '-';
    } else {
        *(iter++) = 'a' + FILENR(pos->ep_sq);
        *(iter++) = '1' + RANKNR(pos->ep_sq);
    }
    *(iter++) = ' ';

    /* Halfmove clock */
    sprintf(iter, "%d", pos->fifty);
    iter += strlen(iter);
    *(iter++) = ' ';

    /* Fullmove number */
    sprintf(iter, "%d", pos->fullmove);

}

void move2str(uint32_t move, struct position *pos, char *str)
{
    int from;
    int to;
    int promotion;

    assert(str != NULL);

    from = FROM(move);
    to = to_square(move);
    promotion = PROMOTION(move);

    if (ISNULLMOVE(move)) {
        strcpy(str, "0000");
        return;
    } else if (move == NOMOVE) {
        strcpy(str, "(none)");
        return;
    } else if (pos->castle == 0) {
        if (ISKINGSIDECASTLE(move)) {
            strcpy(str, "O-O");
            return;
        } else if (ISQUEENSIDECASTLE(move)) {
            strcpy(str, "O-O-O");
            return;
        }
    }

    str[0] = FILENR(from) + 'a';
    str[1] = RANKNR(from) + '1';
    str[2] = FILENR(to) + 'a';
    str[3] = RANKNR(to) + '1';
    if (ISPROMOTION(move)) {
        switch (PIECE_TYPE(promotion)) {
        case KNIGHT:
            str[4] = 'n';
            break;
        case BISHOP:
            str[4] = 'b';
            break;
        case ROOK:
            str[4] = 'r';
            break;
        case QUEEN:
            str[4] = 'q';
            break;
        default:
            assert(false);
            break;
        }
        str[5] = '\0';
    } else {
        str[4] = '\0';
    }
}
