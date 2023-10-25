#!/bin/bash

set -eux

BALLOT_FILE=testdata/ballots/random/rand_hypergeometric.blt
ARITHMETIC=fixed9

RUSTFLAGS='-C force-frame-pointers=y' cargo +nightly build --release
BINARY=./target/release/stv-rs
sleep 5

function record() {
    local EVENT=$1
    local INTERVAL=$2
    local NUM_THREADS=4
    local TRACE_FILE=perf-record.${NUM_THREADS}-threads.${EVENT}

    sleep 1
    RAYON_NUM_THREADS=${NUM_THREADS} perf record -e "${EVENT}" -c "${INTERVAL}" -g "--output=${TRACE_FILE}.perf" \
        "${BINARY}" --arithmetic "${ARITHMETIC}" --input "${BALLOT_FILE}" meek --parallel=true > /dev/null
    perf script "--input=${TRACE_FILE}.perf" -F +pid > "${TRACE_FILE}.processed.perf"
}

record branch-misses 100
record L1-dcache-load-misses 100
record L1-icache-load-misses 100
record LLC-loads 10
record LLC-load-misses 1
