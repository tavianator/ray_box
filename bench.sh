#!/bin/bash

set -eu

COUNT=$((10 ** 10))
IMPLS=(baseline {in,ex}clusive)
THREADS=(1 2 4 6 8 12 16 24 32 48)
LEVELS=({4..10})

sudo cpupower frequency-set -g performance >/dev/null
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null

printf '{\n'

for impl in ${IMPLS[@]}; do
    if [ $impl != ${IMPLS[0]} ]; then
        printf ',\n'
    fi
    printf '    "%s": [\n' "$impl"

    for threads in ${THREADS[@]}; do
        if [ $threads -ne ${THREADS[0]} ]; then
            printf ',\n'
        fi
        printf '        ['

        for levels in ${LEVELS[@]}; do
            if [ $levels -ne ${LEVELS[0]} ]; then
                printf ', '
            fi
            ./ray_box_$impl $COUNT $levels $threads
        done

        printf ']'
    done

    printf '\n    ]'
done

printf '\n}\n'

sudo cpupower frequency-set -g schedutil >/dev/null
echo 1 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null
