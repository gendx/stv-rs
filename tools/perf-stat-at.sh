#!/bin/bash

set -eux

BALLOT_FILE=testdata/ballots/random/rand_hypergeometric.blt
ARITHMETIC=fixed9
REPETITIONS=100
#ARITHMETIC=bigfixed9
#REPETITIONS=10

INDEX=$1
COMMIT=$2
BINARY=./bin/stv-rs-${INDEX}-${COMMIT}

if [ ! -e "${BINARY}" ]; then
    echo "Binary not found at ${BINARY}. Please build it via the script at 'tools/perf-all-commits.sh'."
    exit 42
fi

SLEEP_SECONDS=5

function sample() {
    local ARITHMETIC=$1
    local TITLE=$2
    local PARAMS
    read -r -a PARAMS <<< "$3"

    set +x
    echo "****************************************"
    echo "* ${TITLE} *"
    echo "****************************************"
    set -x

    sleep "${SLEEP_SECONDS}"
    perf stat -r "${REPETITIONS}" "${PARAMS[@]}" \
        "${BINARY}" --arithmetic "${ARITHMETIC}" --input "${BALLOT_FILE}" meek --parallel=false > /dev/null

    for NUM_THREADS in 2 4 8
    do
        sleep "${SLEEP_SECONDS}"
        RAYON_NUM_THREADS=${NUM_THREADS} perf stat -r "${REPETITIONS}" "${PARAMS[@]}" \
            "${BINARY}" --arithmetic "${ARITHMETIC}" --input "${BALLOT_FILE}" meek --parallel=true > /dev/null
    done
}

function sample_default() {
    sample "$1" "STATS" "-d -d"
}

function sample_events() {
    sample "$1" "$2" "-e $3"
}

sample_default fixed9
sample_events fixed9 "INSTRUCTIONS" "task-clock,user_time,system_time,duration_time,cycles:u,instructions:u,branch-instructions:u,branch-misses:u"
sample_events fixed9 "L1-CACHE" "L1-dcache-loads,L1-dcache-load-misses,L1-icache-load-misses,LLC-loads"
sample_events fixed9 "LL-CACHE" "L1-dcache-loads,LLC-loads,LLC-load-misses"
#sample_events fixed9 "STORES" "L1-dcache-stores,LLC-stores,LLC-store-misses"
