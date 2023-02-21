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

#if __linux__
#define _GNU_SOURCE
#endif

#include "black_box.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <string.h>
#include <time.h>

#if SIMD

#if __AVX2__

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
#if INCLUSIVE || SIGNS
#define CMP _CMP_LE_OQ
#else
#define CMP _CMP_LT_OQ
#endif
    vfloat mask = _mm256_cmp_ps(tmin, tmax, CMP);
    return _mm256_blendv_ps(t, tmin, mask);
}

#elif __SSE__

#include <xmmintrin.h>
#include <smmintrin.h>

typedef __m128 vfloat;

static inline vfloat broadcast(float x) {
    return _mm_set1_ps(x);
}

static inline vfloat min(vfloat x, vfloat y) {
    return _mm_min_ps(x, y);
}

static inline vfloat max(vfloat x, vfloat y) {
    return _mm_max_ps(x, y);
}

static inline vfloat newt(vfloat tmin, vfloat tmax, vfloat t) {
#if INCLUSIVE || SIGNS
    vfloat mask = _mm_cmple_ps(tmin, tmax);
#else
    vfloat mask = _mm_cmplt_ps(tmin, tmax);
#endif

#if __SSE4_1__
    return _mm_blendv_ps(t, tmin, mask);
#else
    tmin = _mm_and_ps(mask, tmin);
    t = _mm_andnot_ps(mask, t);
    return _mm_or_ps(tmin, t);
#endif
}

#elif __ARM_NEON

#include <arm_neon.h>

typedef float32x4_t vfloat;

static inline vfloat broadcast(float x) {
    return vdupq_n_f32(x);
}

static inline vfloat min(vfloat x, vfloat y) {
    return vbslq_f32(vcltq_f32(x, y), x, y);
}

static inline vfloat max(vfloat x, vfloat y) {
    return vbslq_f32(vcgtq_f32(x, y), x, y);
}

static inline vfloat newt(vfloat tmin, vfloat tmax, vfloat t) {
    return vbslq_f32(vcltq_f32(tmin, tmax), tmin, t);
}

#else
#error "Which vector instruction set?"
#endif

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
#if INCLUSIVE || SIGNS
    return tmin <= tmax ? tmin : t;
#else
    return tmin < tmax ? tmin : t;
#endif
}

#endif // !SIMD

#define VSIZE (sizeof(vfloat) / sizeof(float))

struct ray {
    float origin[3];
    float dir_inv[3];
};

struct box {
    union {
        float corners[2][3];
        struct {
            float min[3];
            float max[3];
        };
    };
};

#if SIMD
struct vbox {
    union {
        vfloat corners[2][3];
        struct {
            vfloat min[3];
            vfloat max[3];
        };
    };
};

typedef struct vbox vbox;
#else
typedef struct box vbox;
#endif

static void intersections(
    const struct ray *ray,
    size_t nboxes,
    const vbox boxes[nboxes],
    vfloat ts[nboxes])
{
    vfloat origin[3], dir_inv[3];
    for (int d = 0; d < 3; ++d) {
        origin[d] = broadcast(ray->origin[d]);
        dir_inv[d] = broadcast(ray->dir_inv[d]);
    }

#if SIGNS
    bool signs[3];
    for (int d = 0; d < 3; ++d) {
        signs[d] = signbit(ray->dir_inv[d]);
    }
#endif

    for (size_t i = 0; i < nboxes; ++i) {
        const vbox *box = &boxes[i];
        vfloat tmin = broadcast(0.0);
        vfloat tmax = ts[i];

        for (int d = 0; d < 3; ++d) {
#if SIGNS
            vfloat bmin = box->corners[signs[d]][d];
            vfloat bmax = box->corners[!signs[d]][d];
#else
            vfloat bmin = box->min[d];
            vfloat bmax = box->max[d];
#endif

            vfloat t1 = (bmin - origin[d]) * dir_inv[d];
            vfloat t2 = (bmax - origin[d]) * dir_inv[d];

#if BASELINE
            tmin = max(tmin, min(t1, t2));
            tmax = min(tmax, max(t1, t2));
#elif EXCLUSIVE
            tmin = max(tmin, min(min(t1, t2), tmax));
            tmax = min(tmax, max(max(t1, t2), tmin));
#elif INCLUSIVE
            tmin = min(max(t1, tmin), max(t2, tmin));
            tmax = max(min(t1, tmax), min(t2, tmax));
#elif SIGNS
            tmin = max(t1, tmin);
            tmax = min(t2, tmax);
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

#define MALLOC(type, count) xmemalign(alignof(type), (count) * sizeof(type))

static struct box *octree(const struct box *parent, struct box *children, int level) {
    struct box *child = children;

    if (level > 0) {
        for (int i = 0; i < (1 << 3); ++i) {
            for (int d = 0; d < 3; ++d) {
                float mid = (parent->min[d] + parent->max[d]) / 2.0;
                if (i & (1 << d)) {
                    child->min[d] = mid;
                    child->max[d] = parent->max[d];
                } else {
                    child->min[d] = parent->min[d];
                    child->max[d] = mid;
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

#if SIMD
static vbox *pack_boxes(size_t nboxes, size_t *nvboxes, struct box boxes[nboxes]) {
    *nvboxes = (nboxes + VSIZE - 1) / VSIZE;
    vbox *vboxes = MALLOC(vbox, *nvboxes);

    for (size_t i = 0; i < *nvboxes; ++i) {
        for (size_t j = 0, k = i * VSIZE; j < VSIZE; ++j, ++k) {
            if (k == nboxes) {
                --k;
            }
            for (int d = 0; d < 3; ++d) {
                vboxes[i].min[d][j] = boxes[k].min[d];
                vboxes[i].max[d][j] = boxes[k].max[d];
            }
        }
    }

    return vboxes;
}

static void free_vboxes(vbox *vboxes) {
    free(vboxes);
}
#else
static vbox *pack_boxes(size_t nboxes, size_t *nvboxes, struct box boxes[nboxes]) {
    *nvboxes = nboxes;
    return boxes;
}

static void free_vboxes(vbox *vboxes) {
}
#endif

static void reference_impl(
    const struct ray *ray,
    size_t nboxes,
    const struct box boxes[nboxes],
    float ts[nboxes])
{
    for (size_t i = 0; i < nboxes; ++i) {
        const struct box *box = &boxes[i];
        float tmin = 0.0;
        float tmax = ts[i];

        for (int d = 0; d < 3; ++d) {
            if (isfinite(ray->dir_inv[d])) {
                float t1 = (box->min[d] - ray->origin[d]) * ray->dir_inv[d];
                float t2 = (box->max[d] - ray->origin[d]) * ray->dir_inv[d];

                if (t1 < t2) {
                    tmin = tmin > t1 ? tmin : t1;
                    tmax = tmax < t2 ? tmax : t2;
                } else {
                    tmin = tmin > t2 ? tmin : t2;
                    tmax = tmax < t1 ? tmax : t1;
                }
#if INCLUSIVE || SIGNS
            } else if (ray->origin[d] < box->min[d] || ray->origin[d] > box->max[d]) {
#else
            } else if (ray->origin[d] <= box->min[d] || ray->origin[d] >= box->max[d]) {
#endif
                tmin = INFINITY;
                break;
            }
        }

#if INCLUSIVE || SIGNS
        ts[i] = tmin <= tmax ? tmin : ts[i];
#else
        ts[i] = tmin < tmax ? tmin : ts[i];
#endif
    }
}

static void check_ray(const struct ray *ray) {
    // Check boxes with every corner ± 0.01
    size_t nboxes = 3 * 3 * 3;
    nboxes *= nboxes;
    struct box *boxes = MALLOC(struct box, nboxes);

    for (int i = 0; i < nboxes; ++i) {
        int n = i;

        for (int d = 0; d < 3; ++d) {
            boxes[i].min[d] = -1.0 + (n % 3 - 1) * 0.01;
            n /= 3;

            boxes[i].max[d] = +1.0 + (n % 3 - 1) * 0.01;
            n /= 3;
        }
    }

    float *ts = MALLOC(float, nboxes);
    for (size_t i = 0; i < nboxes; ++i) {
        ts[i] = INFINITY;
    }
    reference_impl(ray, nboxes, boxes, ts);

    size_t nvboxes;
    vbox *vboxes = pack_boxes(nboxes, &nvboxes, boxes);

    vfloat *vts = MALLOC(vfloat, nvboxes);
    for (size_t i = 0; i < nvboxes; ++i) {
        vts[i] = broadcast(INFINITY);
    }
    intersections(ray, nvboxes, vboxes, vts);

    for (size_t i = 0; i < nboxes; ++i) {
#if BASELINE
        bool skip = false;
        for (int d = 0; d < 3; ++d) {
            float t1 = (boxes[i].min[d] - ray->origin[d]) * ray->dir_inv[d];
            float t2 = (boxes[i].max[d] - ray->origin[d]) * ray->dir_inv[d];
            if (isnan(t1) || isnan(t2)) {
                skip = true;
                break;
            }
        }
        if (skip) {
            continue;
        }
#endif

        float t = ts[i];
#if SIMD
        float vt = vts[i / VSIZE][i % VSIZE];
#else
        float vt = vts[i];
#endif

        if (!(t == vt || (isnan(t) && isnan(vt)))) {
            printf("ray->origin\t= {%f, %f, %f}\n", ray->origin[0], ray->origin[1], ray->origin[2]);
            printf("ray->dir_inv\t= {%f, %f, %f}\n", ray->dir_inv[0], ray->dir_inv[1], ray->dir_inv[2]);
            printf("boxes[%zu].min\t= {%f, %f, %f}\n", i, boxes[i].min[0], boxes[i].min[1], boxes[i].min[2]);
            printf("boxex[%zu].max\t= {%f, %f, %f}\n", i, boxes[i].max[0], boxes[i].max[1], boxes[i].max[2]);
            printf("t\t\t= %f\n", t);
            printf("vt\t\t= %f\n", vt);
            abort();
        }
    }

    free(vts);
    free_vboxes(vboxes);
    free(ts);
    free(boxes);
}

static void check() {
    // Check rays originating at every corner of a box
    for (int i = 0; i < (1 << 3); ++i) {
        float origin[3];
        for (int d = 0; d < 3; ++d) {
            origin[d] = (i & (1 << d)) ? +1.0 : -1.0;
        }

        // Check rays aiming at every other corner of the box
        for (int j = 0; j < (1 << 3); ++j) {
            if (j == i) {
                continue;
            }

            struct ray ray;
            for (int d = 0; d < 3; ++d) {
                float target = (j & (1 << d)) ? +1.0 : -1.0;
                float delta = target - origin[d];
                // Back up the ray so it doesn't start on the box
                ray.origin[d] = origin[d] - delta;
                ray.dir_inv[d] = 1.0 / delta;
            }

            check_ray(&ray);
        }
    }
}

static void barrier(pthread_barrier_t *barrier, struct timespec *ts) {
    int ret = pthread_barrier_wait(barrier);
    if (ret == PTHREAD_BARRIER_SERIAL_THREAD) {
        if (clock_gettime(CLOCK_MONOTONIC, ts) != 0) {
            die("clock_gettime()", errno);
        }
    } else if (ret != 0) {
        die("pthread_barrier_wait()", ret);
    }
}

struct args {
    pthread_barrier_t *barrier;
    struct timespec *start;
    struct timespec *end;
    size_t niters;
    const struct ray *ray;
    size_t nboxes;
    const vbox *boxes;
};

static void *work(void *ptr) {
    struct args *args = ptr;

    size_t nboxes = args->nboxes;
    size_t chunk = (64 << 10) / sizeof(vfloat);
    chunk = nboxes < chunk ? nboxes : chunk;

    vfloat *ts = MALLOC(vfloat, chunk);
    for (size_t i = 0; i < chunk; ++i) {
        ts[i] = broadcast(INFINITY);
    }
    black_box(ts);

    barrier(args->barrier, args->start);

    for (size_t i = 0; i < args->niters; ++i) {
        size_t j;
        for (j = 0; j + chunk < nboxes; j += chunk) {
            intersections(args->ray, chunk, args->boxes + j, ts);
            black_box(ts);
        }
        intersections(args->ray, nboxes - j, args->boxes + j, ts);
        black_box(ts);
    }

    barrier(args->barrier, args->end);

    free(ts);
    return NULL;
}

#if __linux__
static size_t cpu_next(size_t cpu, cpu_set_t *cpus) {
    while (!CPU_ISSET(cpu, cpus)) {
        ++cpu;
    }
    return cpu;
}

static void cpu_one_hot(size_t cpu, cpu_set_t *cpus) {
    CPU_ZERO(cpus);
    CPU_SET(cpu, cpus);
}
#endif

int main(int argc, char *argv[]) {
    if (argc == 2 && strcmp(argv[1], "check") == 0) {
        check();
        return EXIT_SUCCESS;
    } else if (argc != 4) {
        fprintf(stderr, "Usage: %s <COUNT> <LEVELS> <THREADS>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t count = strtoull(argv[1], NULL, 0);
    size_t levels = strtoull(argv[2], NULL, 0);
    size_t nthreads = strtoull(argv[3], NULL, 0);
    if (nthreads < 1) {
        fprintf(stderr, "Not enough threads\n");
        return EXIT_FAILURE;
    }

    size_t nboxes = (1ULL << (3 * levels)) / 7;
    size_t niters = count / nboxes;
    niters = niters > 0 ? niters : 1;

    struct ray ray = {
        .origin = {-2.0, -2.0, -2.0},
        .dir_inv = {1.0, 1.0, 1.0},
    };
    black_box(&ray);

    struct box *boxes = MALLOC(struct box, nboxes);
    boxes[0] = (struct box) {
        .corners = {
            {-1.0, -1.0, -1.0},
            {+1.0, +1.0, +1.0},
        },
    };
    octree(boxes, boxes + 1, levels - 1);
    black_box(boxes);

    size_t nvboxes;
    vbox *vboxes = pack_boxes(nboxes, &nvboxes, boxes);

    pthread_barrier_t barrier;
    int err = pthread_barrier_init(&barrier, NULL, nthreads);
    if (err != 0) {
        die("pthread_barrier_init()", err);
    }

    struct timespec start, end;

    struct args args = {
        .barrier = &barrier,
        .start = &start,
        .end = &end,
        .niters = niters,
        .ray = &ray,
        .nboxes = nvboxes,
        .boxes = vboxes,
    };

#if __linux__
    cpu_set_t cpus;
    err = pthread_getaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    if (err != 0) {
        die("pthread_getaffinity_np()", err);
    }
    size_t ncpus = CPU_COUNT(&cpus);
    size_t cpu = cpu_next(0, &cpus);

    // If the number of available CPUs and threads are the same, pin each
    // thread to a separate CPU
    if (nthreads == ncpus) {
        cpu_set_t affinity;
        cpu_one_hot(cpu, &affinity);
        err = pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
        if (err != 0) {
            die("pthread_setaffinity_np()", err);
        }
    }
#endif

    pthread_attr_t attrs;
    err = pthread_attr_init(&attrs);
    if (err != 0) {
        die("pthread_attr_init()", err);
    }

    pthread_t *threads = MALLOC(pthread_t, nthreads - 1);
    for (size_t i = 0; i < nthreads - 1; ++i) {
#if __linux__
        if (nthreads == ncpus) {
            cpu = cpu_next(cpu + 1, &cpus);

            cpu_set_t affinity;
            cpu_one_hot(cpu, &affinity);
            err = pthread_attr_setaffinity_np(&attrs, sizeof(affinity), &affinity);
            if (err != 0) {
                die("pthread_attr_setaffinity_np()", err);
            }
        }
#endif

        err = pthread_create(&threads[i], &attrs, work, &args);
        if (err != 0) {
            die("pthread_create()", err);
        }
    }

    err = pthread_attr_destroy(&attrs);
    if (err != 0) {
        die("pthread_attr_destroy()", err);
    }

    work(&args);

    for (size_t i = 0; i < nthreads - 1; ++i) {
        err = pthread_join(threads[i], NULL);
        if (err != 0) {
            die("pthread_join()", err);
        }
    }

    double elapsed = end.tv_sec - start.tv_sec;
    elapsed += 1.0e-9 * (end.tv_nsec - start.tv_nsec);

    printf("%f", nthreads * niters * nvboxes * VSIZE / elapsed / 1.0e9);

    free(threads);
    free_vboxes(vboxes);
    free(boxes);
    return EXIT_SUCCESS;
}
