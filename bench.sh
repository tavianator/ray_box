#!/usr/bin/env tailfin

setup() {
    if ((UID != 0)); then
        printf 'warning: not running as root, benchmarks will be less stable\n'
        return 0
    fi

    if [ "$HOSTNAME" = tachyon ]; then
        # Machine-specific setup (stop services, VMs, etc.)
        source ./tachyon.sh
    fi

    aslr-off
    turbo-off
    max-freq
}

bench() {
    IMPLS=(ray_{,v}box_{baseline,{in,ex}clusive,signs})

    printf '{\n'

    for impl in ${IMPLS[@]}; do
        if [ $impl != ${IMPLS[0]} ]; then
            printf ',\n'
        fi
        printf '    "%s": ' "$impl"

        bench_impl
    done

    printf '\n}\n'
}

bench_impl() {
    case $impl in
        ray_vbox_signs)
            THREADS=(1 2 3 4 6 8 12 16 24 32 48)
            ;;
        *)
            THREADS=(1)
            ;;
    esac

    printf '[\n'

    for threads in ${THREADS[@]}; do
        if ((threads != THREADS[0])); then
            printf ',\n'
        fi

        bench_threads
    done

    printf '\n'
    printf '%4s]' ""
}

bench_threads() {
    DEPTHS=({4..10})

    printf '%8s{\n' ""
    printf '%8s    "threads": %d,\n' "" $threads
    printf '%8s    "depths": [\n' ""

    for depth in ${DEPTHS[@]}; do
        if ((depth != DEPTHS[0])); then
            printf ',\n'
        fi

        bench_depth
    done

    printf '\n'
    printf '%8s    ]\n' ""
    printf '%8s}' ""
}

bench_depth() {
    COUNT=$((10 ** 10))

    printf '%16s{ "depth": %2d, "samples": [' "" $depth

    # Depths >= 8 are much slower, so collect fewer samples
    if ((depth >= 8)); then
        samples=3
    else
        samples=7
        # Discard one warmup run
        bench_sample >/dev/null
    fi

    for sample in $(seq $samples); do
        if ((sample != 1)); then
            printf ', '
        fi
        bench_sample
    done

    printf '] }'
}

bench_sample() {
    cmd=(./$impl $COUNT $depth $threads)

    if [ "$HOSTNAME" = tachyon ]; then
        # Machine-specific tuning for a Threadripper 3960x with 4 DIMMS in NPS4
        # configuration.  Each CCD has two CCXs with 3 cores/6 threads each, and
        # connects to memory via the I/O die.  Shorter distances across the I/O
        # die correspond to lower latencies and higher bandwidth.  Memory is
        # attached to nodes 1 and 2, while nodes 0 and 3 have only compute.  L3$
        # is shared within each CCX.
        #
        #            [         CCD         ] [         CCD         ]
        #            [ +--CCX--+ +--CCX--+ ] [ +--CCX--+ +--CCX--+ ]
        #      Cores [ | 12-14 | | 15-17 | ] [ | 00-02 | | 03-05 | ]
        #    Threads [ | 36-38 | | 39-41 | ] [ | 24-26 | | 27-29 | ]
        #            [ +-------+ +-------+ ] [ +-------+ +-------+ ]
        #            [        Node 2       ] [        Node 0       ]
        #             __________|_______________________|__________
        # [D] <------[                                             ]    [D]
        # [I]        [                                             ]    [I]
        # [M] [D] <--[                   I/O die                   ]    [M] [D]
        # [M] [I]    [                                             ]--> [M] [I]
        #     [M]    [                                             ]        [M]
        #     [M]    [_____________________________________________]------> [M]
        #                       |                       |
        #            [        Node 3       ] [        Node 1       ]
        #            [ +-------+ +-------+ ] [ +-------+ +-------+ ]
        #      Cores [ | 18-20 | | 21-23 | ] [ | 06-08 | | 09-11 | ]
        #    Threads [ | 42-44 | | 45-47 | ] [ | 30-32 | | 33-35 | ]
        #            [ +--CCX--+ +--CCX--+ ] [ +--CCX--+ +--CCX--+ ]
        #            [         CCD         ] [         CCD         ]
        #
        # See https://en.wikichip.org/wiki/amd/ryzen_threadripper/3960x
        # and https://en.wikichip.org/wiki/amd/packages/socket_strx4

        # The 12 cores in nodes 0/1 have fast access to half the memory.  Prefer
        # them unless we have too many threads, or we'll be memory-bound.
        if ((threads > 12 || depth >= 8)); then
            args=(
                 [1]="-m 1 -C 6"
                 [2]="-i 1-2 -C 6-7"                      # Interleave memory but share L3
                 [3]="-i 1-2 -C 6-7,12"
                 [4]="-i 1-2 -C 6-7,12-13"                # 2 cores/CCX on nodes 1-2
                 [6]="-i 1-2 -C 6-8,12-14"                # 3 cores/CCX on nodes 1-2
                 [8]="-i 1-2 -C 6-7,9-10,12-13,15-16"     # 2 cores/CCX on nodes 1-2
                [12]="-i 1-2 -C 6-17"                     # 3 cores/CCX on nodes 1-2
                [16]="-i 1-2 -C !2,5,8,11,14,17,20,23-47" # 2 cores/CCX on nodes 0-3
                [24]="-i 1-2 -C 0-23"                     # 3 cores/CCX on nodes 0-3
                [32]="-i 1-2"
                [48]="-i 1-2"
            )
        else
            args=(
                 [1]="-m 1 -C 6"
                 [2]="-m 1 -C 6-7"
                 [3]="-m 1 -C 6-8"              # 1 CCX on node 1
                 [4]="-m 1 -C 6-7,9-10"         # 2 cores/CCX on node 1
                 [6]="-m 1 -C 6-11"             # 3 cores/CCX on node 1
                 [8]="-m 1 -C 0-1,3-4,6-7,9-10" # 2 cores/CCX on nodes 0-1
                [12]="-m 1 -C 0-11"             # 3 cores/CCX on nodes 0/1
            )
        fi

        numactl ${args[$threads]} "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}
