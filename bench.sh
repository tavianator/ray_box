#!/bin/bash

set -eu

COUNT=$((10 ** 10))
IMPLS=(baseline {in,ex}clusive)
THREADS=(1 2 4 6 8 12 16 24 32 48)
LEVELS=({4..10})

GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
sudo cpupower frequency-set -g performance >/dev/null

INTEL_BOOST=/sys/devices/system/cpu/intel_pstate/no_turbo
OTHER_BOOST=/sys/devices/system/cpu/cpufreq/boost

if [ -e "$INTEL_BOOST" ]; then
    echo 1 | sudo tee "$INTEL_BOOST" >/dev/null
else
    echo 0 | sudo tee "$OTHER_BOOST" >/dev/null
fi

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

sudo cpupower frequency-set -g "$GOVERNOR" >/dev/null

if [ -e "$INTEL_BOOST" ]; then
    echo 0 | sudo tee "$INTEL_BOOST" >/dev/null
else
    echo 1 | sudo tee "$OTHER_BOOST" >/dev/null
fi
