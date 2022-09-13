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
struct vray {
    vfloat origin[3];
    vfloat dir_inv[3];
};

struct vbox {
    vfloat min[3];
    vfloat max[3];
};

typedef struct vray vray;
typedef struct vbox vbox;
#else
typedef struct ray vray;
typedef struct box vbox;
#endif

static void intersections(
    const vray *ray,
    size_t nboxes,
    const vbox boxes[nboxes],
    vfloat ts[nboxes])
{
    for (size_t i = 0; i < nboxes; ++i) {
        const vbox *box = &boxes[i];
        vfloat tmin = broadcast(0.0);
        vfloat tmax = ts[i];

        for (int j = 0; j < 3; ++j) {
            vfloat t1 = (box->min[j] - ray->origin[j]) * ray->dir_inv[j];
            vfloat t2 = (box->max[j] - ray->origin[j]) * ray->dir_inv[j];

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
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 3; ++j) {
                float mid = (parent->max[j] + parent->min[j]) / 2.0;
                if ((i >> j) & 1) {
                    child->min[j] = mid;
                    child->max[j] = parent->max[j];
                } else {
                    child->min[j] = parent->min[j];
                    child->max[j] = mid;
                }
            }
            ++child;
        }

        for (int i = 0; i < 8; ++i) {
            child = octree(&children[i], child, level - 1);
        }
    }

    return child;
}

static void broadcast_ray(vray *vray, const struct ray *ray) {
    for (int i = 0; i < 3; ++i) {
        vray->origin[i] = broadcast(ray->origin[i]);
        vray->dir_inv[i] = broadcast(ray->dir_inv[i]);
    }
}

#if SIMD
static vbox *pack_boxes(size_t *nboxes, size_t *nvboxes, struct box boxes[*nboxes]) {
    *nvboxes = *nboxes / 8;
    *nboxes = *nvboxes * 8;
    vbox *vboxes = MALLOC(vbox, *nvboxes);

    for (size_t i = 0; i < *nvboxes; ++i) {
        const struct box *unpacked = &boxes[8 * i];
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 8; ++k) {
                vboxes[i].min[j][k] = unpacked[k].min[j];
                vboxes[i].max[j][k] = unpacked[k].max[j];
            }
        }
    }

    return vboxes;
}
#else
static vbox *pack_boxes(size_t *nboxes, size_t *nvboxes, struct box boxes[*nboxes]) {
    *nvboxes = *nboxes;
    return boxes;
}
#endif

struct args {
    size_t niters;
    const vray *ray;
    size_t nboxes;
    const vbox *boxes;
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
    size_t nboxes = (1ULL << (3 * levels)) / 7;
    size_t niters = count / nboxes;
    niters = niters > 0 ? niters : 1;

    struct ray ray = {
        .origin = {-2.0, -2.0, -2.0},
        .dir_inv = {1.0, 1.0, 1.0},
    };
    black_box(&ray);

    vray vray;
    broadcast_ray(&vray, &ray);

    struct box *boxes = MALLOC(struct box, nboxes);
    boxes[0] = (struct box) {
        .min = {-1.0, -1.0, -1.0},
        .max = {+1.0, +1.0, +1.0},
    };
    octree(boxes, boxes + 1, levels - 1);
    black_box(boxes);

    size_t nvboxes;
    vbox *vboxes = pack_boxes(&nboxes, &nvboxes, boxes);

    vfloat *ts = MALLOC(vfloat, nvboxes);

    for (size_t i = 0; i < nvboxes; ++i) {
        ts[i] = broadcast(INFINITY);
    }
    black_box(ts);

    intersections(&vray, nvboxes, vboxes, ts);
    black_box(ts);

    struct args *args = MALLOC(struct args, nthreads);
    for (size_t i = 0; i < nthreads; ++i) {
        vfloat *copy = MALLOC(vfloat, nvboxes);
        memcpy(copy, ts, nvboxes * sizeof(vfloat));

        args[i] = (struct args) {
            .niters = niters,
            .ray = &vray,
            .nboxes = nvboxes,
            .boxes = vboxes,
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

    printf("%f", nthreads * niters * nboxes / elapsed / 1.0e9);

    return EXIT_SUCCESS;
}
