/**
 * Copyright (c) 2022 Tavian Barnes
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "black_box.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <string.h>
#include <time.h>

#if SIMD
#include <xmmintrin.h>
#include <immintrin.h>

typedef __m256 vfloat;

static inline vfloat broadcast(float x) {
    return _mm256_set1_ps(x);
}

static inline vfloat min(vfloat x, vfloat y) {
    return _mm256_min_ps(x, y);
}

static inline vfloat max(vfloat x, vfloat y) {
    return _mm256_max_ps(x, y);
}

static inline vfloat newt(vfloat tmin, vfloat tmax, vfloat t) {
#if INCLUSIVE
#define CMP _CMP_LE_OQ
#else
#define CMP _CMP_LT_OQ
#endif
        vfloat mask = _mm256_cmp_ps(tmin, tmax, CMP);
        return _mm256_blendv_ps(t, tmin, mask);
}

#else // !SIMD

typedef float vfloat;

static inline vfloat broadcast(float x) {
    return x;
}

static inline vfloat min(vfloat x, vfloat y) {
    return x < y ? x : y;
}

static inline vfloat max(vfloat x, vfloat y) {
    return x > y ? x : y;
}

static inline vfloat newt(vfloat tmin, vfloat tmax, vfloat t) {
#if INCLUSIVE
    return tmin <= tmax ? tmin : t;
#else
    return tmin < tmax ? tmin : t;
#endif
}

#endif

struct ray {
    float origin[3];
    float dir_inv[3];
};

struct box {
    float min[3];
    float max[3];
};

#if SIMD
struct vbox {
    vfloat min[3];
    vfloat max[3];
};

typedef struct vbox vbox;
#else
typedef struct box vbox;
#endif

void intersections(
    const struct ray *ray,
    size_t nboxes,
    const vbox boxes[nboxes],
    vfloat ts[nboxes])
{
    for (size_t i = 0; i < nboxes; ++i) {
        const vbox *box = &boxes[i];
        vfloat tmin = broadcast(0.0);
        vfloat tmax = ts[i];

        for (int j = 0; j < 3; ++j) {
            vfloat origin = broadcast(ray->origin[j]);
            vfloat dir_inv = broadcast(ray->dir_inv[j]);

            vfloat t1 = (box->min[j] - origin) * dir_inv;
            vfloat t2 = (box->max[j] - origin) * dir_inv;

#if BASELINE
            tmin = max(tmin, min(t1, t2));
            tmax = min(tmax, max(t1, t2));
#elif EXCLUSIVE
            tmin = max(tmin, min(min(t1, t2), tmax));
            tmax = min(tmax, max(max(t1, t2), tmin));
#elif INCLUSIVE
            tmin = min(max(t1, tmin), max(t2, tmin));
            tmax = max(min(t1, tmax), min(t2, tmax));
#else
#error "Which implementation?"
#endif
        }

        ts[i] = newt(tmin, tmax, ts[i]);
    }
}

static noreturn void die(const char *str, int error) {
    errno = error;
    perror(str);
    abort();
}

static void *xmemalign(size_t align, size_t size) {
    void *ptr = aligned_alloc(align, size);
    if (!ptr) {
        die("aligned_alloc()", errno);
    }
    return ptr;
}

#define MALLOC(type, count) xmemalign(alignof(type), count * sizeof(type))

struct box *octree(const struct box *parent, struct box *children, int level) {
    struct box *child = children;

    if (level > 0) {
        float dx = (parent->max[0] - parent->min[0]) / 2.0;
        float dy = (parent->max[1] - parent->min[1]) / 2.0;
        float dz = (parent->max[2] - parent->min[2]) / 2.0;

        for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
                for (int z = 0; z < 2; ++z) {
                    child->min[0] = parent->min[0] + x * dx;
                    child->min[1] = parent->min[1] + y * dy;
                    child->min[2] = parent->min[2] + z * dz;
                    child->max[0] = child->min[0] + dx;
                    child->max[1] = child->min[1] + dy;
                    child->max[2] = child->min[2] + dz;
                    ++child;
                }
            }
        }

        for (int i = 0; i < 8; ++i) {
            child = octree(&children[i], child, level - 1);
        }
    }

    return child;
}

#if SIMD
struct vbox *pack_boxes(size_t nboxes, const struct box boxes[nboxes]) {
    size_t npacked = nboxes / 8;
    struct vbox *packed = aligned_alloc(alignof(struct vbox), npacked * sizeof(*packed));
    if (!packed) {
        die("aligned_alloc()", errno);
    }

    for (size_t i = 0; i < npacked; ++i) {
        const struct box *unpacked = &boxes[8 * i];
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 8; ++k) {
                packed[i].min[j][k] = unpacked[k].min[j];
                packed[i].max[j][k] = unpacked[k].max[j];
            }
        }
    }

    return packed;
}
#endif

struct args {
    size_t niters;
    const struct ray *ray;
    size_t nboxes;
    vbox *boxes;
    vfloat *ts;
};

static void *work(void *ptr) {
    struct args *args = ptr;

    for (int i = 0; i < args->niters; ++i) {
        intersections(args->ray, args->nboxes, args->boxes, args->ts);
        black_box(args->ts);
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <COUNT> <LEVELS> <THREADS>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t count = strtoull(argv[1], NULL, 0);
    size_t levels = strtoull(argv[2], NULL, 0);
    size_t nthreads = strtoull(argv[3], NULL, 0);
    size_t nunpacked = (1ULL << (3 * levels)) / 7;
    size_t niters = count / nunpacked;
    niters = niters > 0 ? niters : 1;

    struct ray ray = {
        .origin = {-2.0, -2.0, -2.0},
        .dir_inv = {1.0, 1.0, 1.0},
    };
    black_box(&ray);

    struct box *unpacked = MALLOC(struct box, nunpacked);
    unpacked[0] = (struct box) {
        .min = {-1.0, -1.0, -1.0},
        .max = {+1.0, +1.0, +1.0},
    };
    octree(unpacked, unpacked + 1, levels - 1);
    black_box(unpacked);

#if SIMD
    vbox *boxes = pack_boxes(nunpacked, unpacked);
    size_t nboxes = nunpacked / 8;
    nunpacked = nboxes * 8;
#else
    vbox *boxes = unpacked;
    size_t nboxes = nunpacked;
#endif

    vfloat *ts = MALLOC(vfloat, nboxes);

    for (size_t i = 0; i < nboxes; ++i) {
        ts[i] = broadcast(INFINITY);
    }
    black_box(ts);

    intersections(&ray, nboxes, boxes, ts);
    black_box(ts);

    struct args *args = MALLOC(struct args, nthreads);
    for (size_t i = 0; i < nthreads; ++i) {
        vfloat *copy = MALLOC(vfloat, nboxes);
        memcpy(copy, ts, nboxes * sizeof(vfloat));

        args[i] = (struct args) {
            .niters = niters,
            .ray = &ray,
            .nboxes = nboxes,
            .boxes = boxes,
            .ts = copy,
        };
    }

    struct timespec start;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        die("clock_gettime()", errno);
    }

    pthread_t *threads = MALLOC(pthread_t, nthreads);
    for (size_t i = 0; i < nthreads; ++i) {
        int ret = pthread_create(&threads[i], NULL, work, &args[i]);
        if (ret != 0) {
            die("pthread_create()", ret);
        }
    }

    for (size_t i = 0; i < nthreads; ++i) {
        int ret = pthread_join(threads[i], NULL);
        if (ret != 0) {
            die("pthread_join()", ret);
        }
    }

    struct timespec end;
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        die("clock_gettime()", errno);
    }

    double elapsed = end.tv_sec - start.tv_sec;
    elapsed += 1.0e-9 * (end.tv_nsec - start.tv_nsec);

    printf("%f", nthreads * niters * nunpacked / elapsed / 1.0e9);

    return EXIT_SUCCESS;
}
