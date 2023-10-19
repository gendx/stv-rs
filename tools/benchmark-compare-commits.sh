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

BEGIN=main
END=HEAD

echo "[*] Listing commits"
git log ${BEGIN}..${END} --reverse --format=oneline
COMMITS=$(git log ${BEGIN}..${END} --reverse --format=oneline | cut -d' ' -f1)

INDEX_COMMITS=
ITER=0
for COMMIT in ${COMMITS}
do
    ITER=$(expr ${ITER} + 1)
    printf -v INDEX "%02d" ${ITER}
    if [ "${INDEX_COMMITS}" != "" ]; then
        INDEX_COMMITS="${INDEX_COMMITS},"
    fi
    INDEX_COMMITS=${INDEX_COMMITS}${INDEX}-${COMMIT}
done
echo "Commits joined as a hyperfine parameter list: ${INDEX_COMMITS}"

echo "[*] Building at all commits"
./tools/build-all-commits.sh ${COMMITS}
sleep 15

SLEEP_SECONDS=15

function benchmark() {
    ARITHMETIC=$1
    EQUALIZE=$2
    INPUT=$3

    ${HYPERFINE_PATH} \
        --setup "sleep ${SLEEP_SECONDS}" \
        --warmup 1 \
        --parameter-list INDEX_COMMIT ${INDEX_COMMITS} \
        "./bin/stv-rs-{INDEX_COMMIT} --arithmetic ${ARITHMETIC} --input ${INPUT} meek ${EQUALIZE} --parallel=false > /dev/null"

    for NUM_THREADS in 2 4 8
    do
        RAYON_NUM_THREADS=${NUM_THREADS} ${HYPERFINE_PATH} \
            --setup "sleep ${SLEEP_SECONDS}" \
            --warmup 1 \
            --parameter-list INDEX_COMMIT ${INDEX_COMMITS} \
            "./bin/stv-rs-{INDEX_COMMIT} --arithmetic ${ARITHMETIC} --input ${INPUT} meek ${EQUALIZE} --parallel=true > /dev/null"
    done
}

benchmark fixed9    ""          "testdata/shuffle_ballots/rand_sorted_lexicographically.blt"
benchmark fixed9    ""          "testdata/shuffle_ballots/rand_10k_sorted_lexicographically.blt"
benchmark fixed9    --equalize  "testdata/ballots/random/rand_hypergeometric.blt"
benchmark bigfixed9 ""          "testdata/ballots/random/rand_hypergeometric.blt"
