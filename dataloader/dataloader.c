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

#include "types.h"
#include "utils.h"
#include "stream.h"
#include "board.h"

#define NUM_SQ                    64
#define NUM_PT                    12
#define NUM_INPUTS                (NUM_SQ*NUM_PT)
#define MAX_ACTIVE_FEATURES       32

static uint32_t piece2index[NSIDES][NPIECES] = {
    {0*NSQUARES, 1*NSQUARES, 2*NSQUARES, 3*NSQUARES, 4*NSQUARES,
        5*NSQUARES, 6*NSQUARES, 7*NSQUARES, 8*NSQUARES, 9*NSQUARES,
        10*NSQUARES, 11*NSQUARES},
    {1*NSQUARES, 0*NSQUARES, 3*NSQUARES, 2*NSQUARES, 5*NSQUARES,
        4*NSQUARES, 7*NSQUARES, 6*NSQUARES, 9*NSQUARES, 8*NSQUARES,
        11*NSQUARES, 10*NSQUARES}
};

static int cmp_int(const void *p1, const void *p2)
{
    int *pi1 = (int*)p1;
    int *pi2 = (int*)p2;

    if (*pi1 < *pi2) {
        return -1;
    } else if (*pi1 > *pi2) {
        return 1;
    }
    return 0;
}

static int transform_square(int sq, int side)
{
    if (side == BLACK) {
        sq = MIRROR(sq);
    }
    return sq;
}

static int feature_index(int sq, int piece, int side)
{
    sq = transform_square(sq, side);
    return sq + piece2index[side][piece];
}

static void add_features_to_batch(int sample_idx, struct sfen *sfen,
                                  int *counter, int *features, float *values,
                                  int side)
{
    struct position *pos = &sfen->pos;
    uint32_t index;
    int nfeatures;
    int indices[32];
    int k;
    int sq;

    nfeatures = 0;
    for (k=0;k<pos->npieces;k++) {
        sq = pos->pieces[k];
        index = feature_index(sq, pos->board[sq], side);
        indices[nfeatures++] = index;
    }

    qsort(indices, nfeatures, sizeof(int), cmp_int);

    for (k=0;k<nfeatures;k++) {
        index = (*counter)*2;
        features[index] = sample_idx;
        features[index+1] = indices[k];
        values[*counter] = 1.0f;
        (*counter)++;
    }
}

static void add_sample_to_batch(int sample_idx, struct sparse_batch *batch,
                                struct sfen *sfen)
{
    batch->is_white[sample_idx] = (float)(sfen->pos.stm == WHITE);
    batch->outcome[sample_idx] = (float)((sfen->result + 1.0f)/2.0f);
    batch->score[sample_idx] = (float)sfen->score;

    add_features_to_batch(sample_idx, sfen, &batch->num_active_white_features,
                          batch->white, batch->white_values, WHITE);
    add_features_to_batch(sample_idx, sfen, &batch->num_active_black_features,
                          batch->black, batch->black_values, BLACK);
}

EXPORT struct stream* CDECL create_sparse_batch_stream(const char* filename,
                                                       uint64_t nsamples,
                                                       int batch_size)
{
    return stream_create((char*)filename, nsamples, batch_size);
}

EXPORT void CDECL destroy_sparse_batch_stream(struct stream *stream)
{
    stream_destroy(stream);
}

EXPORT struct sparse_batch* CDECL fetch_next_sparse_batch(struct stream *stream)
{
    struct sfen         *samples;
    int                 nsamples;
    int                 nactive;
    struct sparse_batch *batch;
    int                 k;

    /* Check if there are still samples left */
    if (stream->nread == stream->nsamples) {
        return NULL;
    }

    /* Calculate the number of active features */
    nactive = MAX_ACTIVE_FEATURES;

    /* Read samples */
    samples = malloc(stream->batch_size*sizeof(struct sfen));
    nsamples = stream_get_samples(stream, samples);
    assert(nsamples > 0);
    assert(stream->nread <= stream->nsamples);

    /* Create batch */
    batch = malloc(sizeof(struct sparse_batch));
    batch->size = nsamples;
    batch->num_inputs = NUM_INPUTS;
    batch->num_active_white_features = 0;
    batch->num_active_black_features = 0;
    batch->is_white = malloc(sizeof(float)*nsamples);
    batch->outcome = malloc(sizeof(float)*nsamples);
    batch->score = malloc(sizeof(float)*nsamples);
    batch->white = malloc(sizeof(int)*nsamples*nactive*2);
    batch->black = malloc(sizeof(int)*nsamples*nactive*2);
    batch->white_values = malloc(sizeof(float)*nsamples*nactive);
    batch->black_values = malloc(sizeof(float)*nsamples*nactive);
    memset(batch->white, 0, sizeof(int)*nsamples*nactive*2);
    memset(batch->black, 0, sizeof(int)*nsamples*nactive*2);

    /* Add all samples to the batch  */
    for (k=0;k<nsamples;k++) {
        add_sample_to_batch(k, batch, &samples[k]);
    }
    free(samples);

    return batch;
}

EXPORT void CDECL destroy_sparse_batch(struct sparse_batch *batch)
{
    if (batch == NULL) {
        return;
    }

    free(batch->is_white);
    free(batch->outcome);
    free(batch->score);
    free(batch->white);
    free(batch->black);
    free(batch->white_values);
    free(batch->black_values);
    free(batch);
}

int main(int argc, char *argv[])
{
    struct stream       *stream;
    struct sparse_batch *batch;
    struct sfen         *samples;
    int                 nsamples;
    int                 k;
    char                fenstr[FEN_MAX_LENGTH];
    char                movestr[6];

    if (argc != 4) {
        printf("Wrong number of arguments\n");
        return 1;
    }

    if (!strcmp(argv[1], "-t")) {
        stream = create_sparse_batch_stream(argv[2], atoi(argv[3]), 8192);
        for (k=0;k<1000;k++) {
            batch = fetch_next_sparse_batch(stream);
            destroy_sparse_batch(batch);
        }
        destroy_sparse_batch_stream(stream);
    } else if (!strcmp(argv[1], "-d")) {
        samples = malloc(8192*sizeof(struct sfen));
        stream = stream_create(argv[2], atoi(argv[3]), 8192);
        do {
            nsamples = stream_get_samples(stream, samples);
            for (k=0;k<nsamples;k++) {
                pos2str(&samples[k].pos, fenstr);
                move2str(samples[k].move, &samples[k].pos, movestr);
                printf("fen %s\n", fenstr);
                printf("move %s\n", movestr);
                printf("score %d\n", samples[k].score);
                printf("ply %d\n", samples[k].ply);
                printf("result %d\n", samples[k].result);
                printf("e\n");
            }
        } while (nsamples == 8192);
        stream_destroy(stream);
        free(samples);
    }

    return 0;
}
