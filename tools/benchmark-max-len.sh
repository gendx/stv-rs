#!/bin/bash

set -eux

if [ -e "${HOME}/.cargo/bin/hyperfine" ]; then
    HYPERFINE_PATH=${HOME}/.cargo/bin/hyperfine
elif [ -e "/usr/bin/hyperfine" ]; then
    HYPERFINE_PATH=/usr/bin/hyperfine
else
    echo "Hyperfine is not installed. Please install it with 'apt install hyperfine' or 'cargo install hyperfine'."
    exit 42
fi

./tools/check-inputs.sh

cargo build --release
sleep 15

BINARY=./target/release/stv-rs
if [ ! -e "${BINARY}" ]; then
    echo "Binary not found at ${BINARY}."
    exit 42
fi

SLEEP_SECONDS=15

function benchmark() {
    local ARITHMETIC=$1
    local EQUALIZE=$2
    local INPUT=$3
    local NUM_THREADS=4

    "${HYPERFINE_PATH}" \
        --style color \
        --sort command \
        --setup "sleep ${SLEEP_SECONDS}" \
        --warmup 1 \
        --parameter-list PARALLEL '--parallel=rayon --max-serial-len=1','--parallel=rayon --max-serial-len=2','--parallel=rayon --max-serial-len=4','--parallel=rayon --max-serial-len=8','--parallel=rayon --max-serial-len=16','--parallel=rayon --max-serial-len=32','--parallel=rayon --max-serial-len=64','--parallel=rayon --max-serial-len=128','--parallel=rayon','--parallel=custom' \
        "${BINARY} --arithmetic ${ARITHMETIC} --input ${INPUT} meek ${EQUALIZE} --num-threads=${NUM_THREADS} {PARALLEL} > /dev/null"
}

benchmark bigfixed9 ""          "testdata/ballots/random/rand_2x10.blt"
benchmark fixed9    ""          "testdata/ballots/skewed.blt"
benchmark fixed9    --equalize  "testdata/ballots/skewed.blt"
benchmark fixed9    ""          "testdata/ballots/random/rand_mixed_5k.blt"
benchmark fixed9    --equalize  "testdata/ballots/random/rand_mixed_5k.blt"
benchmark bigfixed9 ""          "testdata/ballots/random/rand_mixed_5k.blt"
benchmark bigfixed9 --equalize  "testdata/ballots/random/rand_mixed_5k.blt"
benchmark fixed9    ""          "testdata/ballots/random/rand_hypergeometric_10k.blt"
benchmark fixed9    --equalize  "testdata/ballots/random/rand_hypergeometric_10k.blt"
