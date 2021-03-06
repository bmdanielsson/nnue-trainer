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

#define NUM_SQ              64
#define NUM_PT              10
#define NUM_PLANES          (NUM_SQ * NUM_PT + 1)
#define NUM_INPUTS          (NUM_PLANES * NUM_SQ)
#define MAX_ACTIVE_FEATURES 32

static uint32_t piece2index[NSIDES][NPIECES] = {
    {0*NSQUARES+1, 1*NSQUARES+1, 2*NSQUARES+1, 3*NSQUARES+1, 4*NSQUARES+1,
        5*NSQUARES+1, 6*NSQUARES+1, 7*NSQUARES+1, 8*NSQUARES+1, 9*NSQUARES+1,
        10*NSQUARES+1, 11*NSQUARES+1},
    {1*NSQUARES+1, 0*NSQUARES+1, 3*NSQUARES+1, 2*NSQUARES+1, 5*NSQUARES+1,
        4*NSQUARES+1, 7*NSQUARES+1, 6*NSQUARES+1, 9*NSQUARES+1, 8*NSQUARES+1,
        11*NSQUARES+1, 10*NSQUARES+1}
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
    /* For black the board is rotated 180 degrees */
    if (side == BLACK) {
        sq = SQUARE(7-FILENR(sq), 7-RANKNR(sq));
    }
    return sq;
}

static int feature_index(int sq, int piece, int king_sq, int side)
{
    sq = transform_square(sq, side);
    return sq + piece2index[side][piece] + (KING*NSQUARES+1)*king_sq;
}


static void add_features_to_batch(int sample_idx,
                                  struct sfen *sfen, int *counter,
                                  int *features, float *values,
                                  int side)
{
    struct position *pos = &sfen->pos;
    uint32_t index;
    int king_sq;
    int nfeatures;
    int indices[32];
    int k;
    int sq;

    king_sq = transform_square(pos->kings[side], side);

    nfeatures = 0;
    for (k=0;k<pos->npieces;k++) {
        sq = pos->pieces[k];
        index = feature_index(sq, pos->board[sq], king_sq, side);
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

    add_features_to_batch(sample_idx, sfen,
                          &batch->num_active_white_features,
                          batch->white, batch->white_values,
                          WHITE);
    add_features_to_batch(sample_idx, sfen,
                          &batch->num_active_black_features,
                          batch->black, batch->black_values,
                          BLACK);
}

EXPORT struct stream* CDECL create_sparse_batch_stream(const char* filename,
                                                       int batch_size)
{
    return stream_create((char*)filename, batch_size);
}

EXPORT void CDECL destroy_sparse_batch_stream(struct stream *stream)
{
    stream_destroy(stream);
}

EXPORT struct sparse_batch* CDECL fetch_next_sparse_batch(struct stream *stream)
{
    struct sfen         *samples;
    int                 nsamples;
    struct sparse_batch *batch;
    int                 k;

    /* Check if there are still samples left */
    if (stream->nread == stream->nsamples) {
        return NULL;
    }

    /* Read samples */
    samples = malloc(stream->batch_size*sizeof(struct sfen));
    nsamples = stream_get_samples(stream, samples);
    assert(nsamples > 0);
    assert((uint32_t)nsamples <= (stream->nread - stream->nsamples));
    stream->nread += nsamples;

    /* Create batch */
    batch = malloc(sizeof(struct sparse_batch));
    batch->size = nsamples;
    batch->num_inputs = NUM_INPUTS;
    batch->num_active_white_features = 0;
    batch->num_active_black_features = 0;
    batch->is_white = malloc(sizeof(float)*nsamples);
    batch->outcome = malloc(sizeof(float)*nsamples);
    batch->score = malloc(sizeof(float)*nsamples);
    batch->white = malloc(sizeof(int)*nsamples*MAX_ACTIVE_FEATURES*2);
    batch->black = malloc(sizeof(int)*nsamples*MAX_ACTIVE_FEATURES*2);
    batch->white_values = malloc(sizeof(float)*nsamples*MAX_ACTIVE_FEATURES);
    batch->black_values = malloc(sizeof(float)*nsamples*MAX_ACTIVE_FEATURES);
    memset(batch->white, 0, sizeof(int)*nsamples*MAX_ACTIVE_FEATURES*2);
    memset(batch->black, 0, sizeof(int)*nsamples*MAX_ACTIVE_FEATURES*2);

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
    int                 k;

    if (argc != 2) {
        printf("Wrong number of arguments\n");
        return 1;
    }

    stream = create_sparse_batch_stream(argv[1], 8192);
    for (k=0;k<1000;k++) {
        batch = fetch_next_sparse_batch(stream);
        destroy_sparse_batch(batch);
    }
    destroy_sparse_batch_stream(stream);

    return 0;
}
