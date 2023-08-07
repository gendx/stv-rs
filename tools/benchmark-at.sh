#!/bin/bash

set -eux

if [ -e "${HOME}/.cargo/bin/hyperfine" ]; then
    HYPERFINE_PATH=${HOME}/.cargo/bin/hyperfine
elif [ -e "/usr/bin/hyperfine" ]; then
    HYPERFINE_PATH=/usr/bin/hyperfine
else
    echo "Hyperfine is not installed. Please install it with 'apt install hyperfine' or 'cargo +nightly install hyperfine'."
    exit 42
fi

INDEX=$1
COMMIT=$2
BINARY=./bin/stv-rs-${INDEX}-${COMMIT}

if [ ! -e "${BINARY}" ]; then
    echo "Binary not found at ${BINARY}. Please build it via the script at 'tools/benchmark-all-commits.sh'."
    exit 42
fi

SLEEP_SECONDS=10

function benchmark() {
    ARITHMETIC=$1
    EQUALIZE=$2
    INPUT=$3

    sleep ${SLEEP_SECONDS}
    ${HYPERFINE_PATH} --warmup 1 \
        "${BINARY} --arithmetic ${ARITHMETIC} --input ${INPUT} meek ${EQUALIZE} --parallel=false > /dev/null"
    for NUM_THREADS in 2 4 8
    do
        sleep ${SLEEP_SECONDS}
        RAYON_NUM_THREADS=${NUM_THREADS} ${HYPERFINE_PATH} --warmup 1 \
            "${BINARY} --arithmetic ${ARITHMETIC} --input ${INPUT} meek ${EQUALIZE} --parallel=true > /dev/null"
    done
}

benchmark fixed9    ""          "testdata/ballots/random/rand_hypergeometric.blt"
benchmark fixed9    ""          "testdata/shuffle_ballots/rand_sorted_lexicographically.blt"
benchmark fixed9    ""          "testdata/shuffle_ballots/rand_10k_sorted_lexicographically.blt"
benchmark fixed9    --equalize  "testdata/ballots/random/rand_hypergeometric.blt"
#benchmark fixed9    ""          "testdata/shuffle_ballots/rand_sorted_by_product.blt"
#benchmark fixed9    ""          "testdata/shuffle_ballots/rand_sorted_by_lexico_product.blt"
benchmark bigfixed9 ""          "testdata/ballots/random/rand_hypergeometric.blt"
