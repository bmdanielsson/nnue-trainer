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
#ifndef UTILS_H
#define UTILS_H

#ifdef WINDOWS
#include <windows.h>
#else
#include <pthread.h>
#endif
#include <stdint.h>
#include <stdbool.h>

/* Macros for exporting symbols */
#ifdef WINDOWS
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#define CDECL
#endif

/* Portable multi-threading primitives */
#ifdef WINDOWS
#define thread_retval_t DWORD
typedef HANDLE thread_t;
typedef LPTHREAD_START_ROUTINE thread_func_t;
typedef CRITICAL_SECTION mutex_t;
typedef HANDLE event_t;
#else
#define thread_retval_t void*
typedef pthread_t thread_t;
typedef void* (*thread_func_t)(void*);
typedef pthread_mutex_t mutex_t;
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    bool            is_set;
} event_t;
#endif

#define FILE_SIZE_ERROR UINT64_MAX

uint64_t get_file_size(char *file);

void sleep_ms(int ms);

void thread_create(thread_t *thread, thread_func_t func, void *data);

void thread_join(thread_t *thread);

void mutex_init(mutex_t *mutex);

void mutex_destroy(mutex_t *mutex);

void mutex_lock(mutex_t *mutex);

void mutex_unlock(mutex_t *mutex);

void event_init(event_t *event);

void event_destroy(event_t *event);

void event_set(event_t *event);

void event_wait(event_t *event);

#endif
