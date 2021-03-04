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
#include <ctype.h>
#include <assert.h>
#include <string.h>
#if defined(WINDOWS)
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "utils.h"

uint64_t get_file_size(char *file)
{
    assert(file != NULL);

#ifdef WINDOWS
    HANDLE fh;
    LARGE_INTEGER size;

    fh = CreateFile(file, GENERIC_READ, 0, NULL, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL, NULL);
	if (!GetFileSizeEx(fh, &size)) {
		return FILE_SIZE_ERROR;
	}
    CloseHandle(fh);

    return (uint64_t)size.QuadPart;
#else
    struct stat sb;

    if (stat(file, &sb) != 0) {
        return FILE_SIZE_ERROR;
    }
    return (uint64_t)sb.st_size;
#endif
}

void sleep_ms(int ms)
{
#ifdef WINDOWS
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec = ms/1000;
    ts.tv_nsec = (ms%1000)*1000000;
    nanosleep(&ts, NULL);
#endif
}

void thread_create(thread_t *thread, thread_func_t func, void *data)
{
#ifdef WINDOWS
    *thread = CreateThread(NULL, 0, func, data, 0, NULL);
#else
    (void)pthread_create(thread, NULL, func, data);
#endif
}

void thread_join(thread_t *thread)
{
#ifdef WINDOWS
    WaitForSingleObject(*thread, INFINITE);
	CloseHandle(*thread);
#else
    (void)pthread_join(*thread, NULL);
#endif
}

void mutex_init(mutex_t *mutex)
{
#ifdef WINDOWS
    InitializeCriticalSection(mutex);
#else
    (void)pthread_mutex_init(mutex, NULL);
#endif
}

void mutex_destroy(mutex_t *mutex)
{
#ifdef WINDOWS
    DeleteCriticalSection(mutex);
#else
    (void)pthread_mutex_destroy(mutex);
#endif
}

void mutex_lock(mutex_t *mutex)
{
#ifdef WINDOWS
    EnterCriticalSection(mutex);
#else
    (void)pthread_mutex_lock(mutex);
#endif
}

void mutex_unlock(mutex_t *mutex)
{
#ifdef WINDOWS
    LeaveCriticalSection(mutex);
#else
    (void)pthread_mutex_unlock(mutex);
#endif
}

void event_init(event_t *event)
{
#ifdef WINDOWS
    *event = CreateEvent(NULL, FALSE, FALSE, NULL);
	ResetEvent(*event);
#else
    pthread_mutex_init(&event->mutex, NULL);
    pthread_cond_init(&event->cond, NULL);
    event->is_set = false;
#endif
}

void event_destroy(event_t *event)
{
#ifdef WINDOWS
    CloseHandle(*event);
#else
    pthread_mutex_destroy(&event->mutex);
    pthread_cond_destroy(&event->cond);
    event->is_set = false;
#endif
}

void event_set(event_t *event)
{
#ifdef WINDOWS
    SetEvent(*event);
#else
    pthread_mutex_lock(&event->mutex);
    event->is_set = true;
    pthread_cond_signal(&event->cond);
    pthread_mutex_unlock(&event->mutex);
#endif
}

void event_wait(event_t *event)
{
#ifdef WINDOWS
    WaitForSingleObject(*event, INFINITE);
#else
    pthread_mutex_lock(&event->mutex);
    while (!event->is_set) {
        pthread_cond_wait(&event->cond, &event->mutex);
    }
    event->is_set = false;
    pthread_mutex_unlock(&event->mutex);
#endif
}
