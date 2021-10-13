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

/*
 * Array of masks for updating castling permissions. For instance
 * a mask of 13 on A1 means that if a piece is moved to/from this
 * square then WHITE can still castle king side and black can still
 * castle both king side and queen side.
 */
static int castling_permission_masks[NSQUARES] = {
    13, 15, 15, 15, 12, 15, 15, 14,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
     7, 15, 15, 15,  3, 15, 15, 11
};

static void add_piece(struct position *pos, int piece, int square)
{
    int k;

    pos->board[square] = piece;
    if (PIECE_TYPE(piece) == KING) {
        for (k=0;k<NSIDES;k++) {
            if (pos->kings[k] == NO_SQUARE) {
                pos->kings[k] = square;
            }
        }
    } else {
        pos->pieces[pos->npieces++] = square;
    }
}

static void remove_piece(struct position *pos, int piece, int square)
{
    int k;

    pos->board[square] = NO_PIECE;
    if (PIECE_TYPE(piece) == KING) {
        for (k=0;k<NSIDES;k++) {
            if (pos->kings[k] == square) {
                pos->kings[k] = NO_SQUARE;
            }
        }
    } else {
        for (k=0;k<30;k++) {
            if (pos->pieces[k] == square) {
                pos->pieces[k] = pos->pieces[--pos->npieces];
                break;
            }
        }
    }
}

static void move_piece(struct position *pos, int piece, int from, int to)
{
    remove_piece(pos, piece, from);
    add_piece(pos, piece, to);
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

void move2str(uint32_t move, char *str)
{
    int from;
    int to;
    int promotion;

    assert(str != NULL);

    from = FROM(move);
    to = TO(move);
    promotion = PROMOTION(move);

    if (ISNULLMOVE(move)) {
        strcpy(str, "0000");
        return;
    } else if (move == NOMOVE) {
        strcpy(str, "(none)");
        return;
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

void make_move(struct position *pos, uint32_t move)
{
    int capture;
    int piece;
    int from;
    int to;
    int promotion;
    int ep;

    from = FROM(move);
    to = TO(move);
    promotion = PROMOTION(move);

    /* Find the pieces involved in the move */
    capture = pos->board[to];
    piece = pos->board[from];

    /* Check if the move enables an en passant capture */
    if ((PIECE_TYPE(piece) == PAWN) && (abs(to-from) == 16)) {
        pos->ep_sq = (pos->stm == WHITE)?to-8:to+8;
    } else {
        pos->ep_sq = NO_SQUARE;
    }

    /* Update castling availability */
    pos->castle &= castling_permission_masks[from];
    pos->castle &= castling_permission_masks[to];

    /* Remove piece from current position */
    remove_piece(pos, piece, from);

    /* If necessary remove captured piece */
    if (ISCAPTURE(move)) {
        remove_piece(pos, capture, to);
    } else if (ISENPASSANT(move)) {
        ep = (pos->stm == WHITE)?to-8:to+8;
        remove_piece(pos, PAWN+FLIP_COLOR(pos->stm), ep);
    }

    /* Add piece to new position */
    if (ISPROMOTION(move)) {
        add_piece(pos, promotion, to);
    } else {
        add_piece(pos, piece, to);
    }

    /* If this is a castling we have to move the rook */
    if (ISKINGSIDECASTLE(move)) {
        move_piece(pos, pos->stm+ROOK, to+1, to-1);
    } else if (ISQUEENSIDECASTLE(move)) {
        move_piece(pos, pos->stm+ROOK, to-2, to+1);
    }

    /* Update the fifty move draw counter */
    if (ISCAPTURE(move) || (PIECE_TYPE(piece) == PAWN)) {
        pos->fifty = 0;
    } else {
        pos->fifty++;
    }

    /* Update fullmove counter */
    if (pos->stm == BLACK) {
        pos->fullmove++;
    }

    /* Change side to move */
    pos->stm = FLIP_COLOR(pos->stm);
}
