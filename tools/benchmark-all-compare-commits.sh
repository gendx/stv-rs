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

function benchmark() {
    TITLE=$1
    ARITHMETIC=$2
    EQUALIZE=$3
    INPUT=$4

    ./tools/benchmark-compare-commits.sh "${INDEX_COMMITS}" "${ARITHMETIC}" "${EQUALIZE}" "${INPUT}" 2>&1 | tee compare-commits.${TITLE}.log
}

echo "[*] Comparing commits for each scenario"
benchmark "1k-fixed9"          fixed9    ""          "testdata/ballots/random/rand_hypergeometric.blt"
benchmark "1k-lexico-fixed9"   fixed9    ""          "testdata/shuffle_ballots/rand_sorted_lexicographically.blt"
benchmark "10k-lexico-fixed9"  fixed9    ""          "testdata/shuffle_ballots/rand_10k_sorted_lexicographically.blt"
benchmark "1k-equalize-fixed9" fixed9    --equalize  "testdata/ballots/random/rand_hypergeometric.blt"
benchmark "1k-bigfixed9"       bigfixed9 ""          "testdata/ballots/random/rand_hypergeometric.blt"
