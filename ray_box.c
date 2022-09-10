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
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <xmmintrin.h>

static inline float min(float x, float y) {
#if __GNUC__ && !__clang__
    return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(x), _mm_set_ss(y)));
#else
    return x < y ? x : y;
#endif
}

static inline float max(float x, float y) {
#if __GNUC__ && !__clang__
    return _mm_cvtss_f32(_mm_max_ss(_mm_set_ss(x), _mm_set_ss(y)));
#else
    return x > y ? x : y;
#endif
}

struct ray {
    float origin[3];
    float dir_inv[3];
};

struct box {
    float min[3];
    float max[3];
};

void intersections(
    const struct ray *ray,
    size_t nboxes,
    const struct box boxes[nboxes],
    float ts[nboxes])
{
    for (size_t i = 0; i < nboxes; ++i) {
        const struct box *box = &boxes[i];
        float tmin = 0.0, tmax = ts[i];

        for (int j = 0; j < 3; ++j) {
            float t1 = (box->min[j] - ray->origin[j]) * ray->dir_inv[j];
            float t2 = (box->max[j] - ray->origin[j]) * ray->dir_inv[j];

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

#if INCLUSIVE
        ts[i] = tmin <= tmax ? tmin : ts[i];
#else
        ts[i] = tmin < tmax ? tmin : ts[i];
#endif
    }
}

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

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <LEVELS> <COUNT>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t levels = strtoull(argv[1], NULL, 0);
    size_t count = strtoull(argv[2], NULL, 0);
    size_t nboxes = (1ULL << (3 * levels)) / 7;
    size_t niters = count / nboxes;
    niters = niters > 0 ? niters : 1;

    struct ray ray = {
        .origin = {-2.0, -2.0, -2.0},
        .dir_inv = {1.0, 1.0, 1.0},
    };
    black_box(&ray);

    struct box *boxes = malloc(nboxes * sizeof(*boxes));
    if (!boxes) {
        perror("malloc()");
        return EXIT_FAILURE;
    }

    boxes[0] = (struct box) {
        .min = {-1.0, -1.0, -1.0},
        .max = {+1.0, +1.0, +1.0},
    };
    octree(boxes, boxes + 1, levels - 1);
    black_box(boxes);

    float *ts = malloc(nboxes * sizeof(*ts));
    if (!ts) {
        perror("malloc()");
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < nboxes; ++i) {
        ts[i] = INFINITY;
    }
    black_box(ts);

    intersections(&ray, nboxes, boxes, ts);
    black_box(ts);

    struct timespec start;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        perror("clock_gettime()");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < niters; ++i) {
        intersections(&ray, nboxes, boxes, ts);
        black_box(ts);
    }

    struct timespec end;
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        perror("clock_gettime()");
        return EXIT_FAILURE;
    }

    double elapsed = end.tv_sec - start.tv_sec;
    elapsed += 1.0e-9 * (end.tv_nsec - start.tv_nsec);

    printf("%f Mboxes/s (%zu boxes in %f s)\n", niters * nboxes / elapsed / 1.0e6, niters * nboxes, elapsed);

    free(ts);
    free(boxes);
    return EXIT_SUCCESS;
}
