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
#include <string.h>
#include <assert.h>
#include <time.h>

#include "stream.h"

static void fill_worker_queue(struct stream *stream)
{
    uint8_t buffer[SFEN_BIN_SIZE];
    size_t  n;
    int     k;

    mutex_lock(&stream->stream_lock);

    for (k=stream->nentries;
         k < SFEN_BUFFER_SIZE && stream->iter < stream->nsamples;
         k++,stream->iter++) {
        n = fread(buffer, SFEN_BIN_SIZE, 1, stream->fp);
        assert(n == 1);
        sfen_unpack_bin(buffer, &stream->buffer[k]);
    }
    stream->nentries = k;

    mutex_unlock(&stream->stream_lock);
}

static thread_retval_t worker_thread_func(void *data)
{
    struct stream *stream = (struct stream*)data;

    /* Main thread loop */
    while (!stream->exit && (stream->iter < stream->nsamples)) {
        /* Refill queue */
        fill_worker_queue(stream);

        /* Signal stream that there are more entries to read */
        event_set(&stream->read_event);

        /* Sleep until told to wakeup */
        event_wait(&stream->write_event);
    }

    return (thread_retval_t)0;
}

static int read_samples(struct stream *stream, struct sfen *buffer, int to_read)
{
    int count = 0;
    int index;

    mutex_lock(&stream->stream_lock);

    /* Loop until the requested number of samples has been read */
    while ((count < to_read) && (stream->nentries > 0)) {
        /* Select a random entry in the queue */
        index = rand()%stream->nentries;
        buffer[count++] = stream->buffer[index];

        /* Replace the selected entry with the last entry in the queue */
        stream->buffer[index] = stream->buffer[--stream->nentries];
    }

    mutex_unlock(&stream->stream_lock);

    return count;
}

struct stream* stream_create(char *filename, int batch_size)
{
    struct stream *stream;
    uint32_t      size;

    /* Seed RNG */
    srand(time(NULL));

    /* Get the size of the data file */
    size = get_file_size((char*)filename);
    if (size == 0xFFFFFFFF) {
        return NULL;
    }
    if (size%SFEN_BIN_SIZE != 0) {
        return NULL;
    }

    /* Create stream */
    stream = malloc(sizeof(struct stream));
    stream->fp = fopen(filename, "rb");
    stream->iter = 0;
    stream->batch_size = batch_size;
    stream->nsamples = size/SFEN_BIN_SIZE;
    stream->nread = 0;
    mutex_init(&stream->stream_lock);
    event_init(&stream->write_event);
    event_init(&stream->read_event);
    stream->exit = false;
    stream->nentries = 0; 

    /* Start worker thread */
    thread_create(&stream->thread, worker_thread_func, stream);

    return stream;
}

void stream_destroy(struct stream *stream)
{
    if (stream == NULL) {
        return;
    }

    /* Stop worker thread */
    stream->exit = true;
    event_set(&stream->write_event);
    thread_join(&stream->thread);

    /* Clean up */
    mutex_destroy(&stream->stream_lock);
    event_destroy(&stream->write_event);
    event_destroy(&stream->read_event);
    fclose(stream->fp);
    free(stream);
}

int stream_get_samples(struct stream *stream, struct sfen *sfen)
{
    int to_read;
    int count = 0;

    /* Calculate how many samples to read */
    to_read = stream->batch_size;
    if ((stream->nsamples - stream->nread) < (uint32_t)to_read) {
        to_read = stream->nsamples - stream->nread;
    }

    /* Loop until a full batch of samples has been read */
    while (count < to_read) {
        /* Read samples */
        count += read_samples(stream, &sfen[count], to_read-count);

        /* Tell the worker to refill the buffer */
        event_set(&stream->write_event);

        /* If more data is needed then wait for worker */
        if (count < to_read) {
            event_wait(&stream->read_event);
        }
    }
    assert(count == to_read);

    return count;
}