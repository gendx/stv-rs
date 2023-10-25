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

INDEX_COMMITS=$1
ARITHMETIC=$2
EQUALIZE=$3
INPUT=$4

SLEEP_SECONDS=15

"${HYPERFINE_PATH}" \
    --style color \
    --setup "sleep ${SLEEP_SECONDS}" \
    --warmup 1 \
    --parameter-list INDEX_COMMIT "${INDEX_COMMITS}" \
    "./bin/stv-rs-{INDEX_COMMIT} --arithmetic ${ARITHMETIC} --input ${INPUT} meek ${EQUALIZE} --parallel=false > /dev/null"

for NUM_THREADS in 2 4 8
do
    RAYON_NUM_THREADS=${NUM_THREADS} "${HYPERFINE_PATH}" \
        --style color \
        --setup "sleep ${SLEEP_SECONDS}" \
        --warmup 1 \
        --parameter-list INDEX_COMMIT "${INDEX_COMMITS}" \
        "./bin/stv-rs-{INDEX_COMMIT} --arithmetic ${ARITHMETIC} --input ${INPUT} meek ${EQUALIZE} --parallel=true > /dev/null"
done
