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
#ifndef STREAM_H
#define STREAM_H

#include <stdio.h>
#include <stdint.h>

#include "sfen.h"
#include "utils.h"

#define SFEN_BUFFER_SIZE 100000

struct stream {
    FILE *fp;
    uint64_t iter;

    int batch_size;
    bool use_factorizer;
    uint64_t nsamples;
    uint64_t nread;

    thread_t thread;
    mutex_t stream_lock;
    event_t write_event;
    event_t read_event;
    bool exit;

    struct sfen buffer[SFEN_BUFFER_SIZE];
    uint32_t nentries;
};

struct stream* stream_create(char *filename, uint64_t nsamples, int batch_size,
                             bool use_factorizer);

void stream_destroy(struct stream *stream);

int stream_get_samples(struct stream *stream, struct sfen *sfen);

#endif
