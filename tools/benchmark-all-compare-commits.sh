#!/bin/bash

set -eux

if [ ! -e "${HOME}/.cargo/bin/hyperfine" ] && [ ! -e "/usr/bin/hyperfine" ]; then
    echo "Hyperfine is not installed. Please install it with 'apt install hyperfine' or 'cargo +nightly install hyperfine'."
    exit 42
fi

function build_commits() {
    local BEGIN=$1
    local END=$2

    echo "[*] Listing commits"
    git log "${BEGIN}..${END}" --reverse --format=oneline
    mapfile -t COMMITS < <(git log "${BEGIN}..${END}" --reverse --format=oneline | cut -d' ' -f1)

    echo "[*] Building at all commits"
    ./tools/build-all-commits.sh "${COMMITS[@]}"
}

function index_commits() {
    local COMMITS=( "$@" )

    INDEX_COMMITS=
    local ITER=0
    for COMMIT in "${COMMITS[@]}"
    do
        ITER=$((ITER + 1))
        printf -v INDEX "%02d" "${ITER}"
        if [ "${INDEX_COMMITS}" != "" ]; then
            INDEX_COMMITS="${INDEX_COMMITS},"
        fi
        INDEX_COMMITS=${INDEX_COMMITS}${INDEX}-${COMMIT}
    done
    echo "Commits joined as a hyperfine parameter list: ${INDEX_COMMITS}"
}

function benchmark() {
    local INDEX_COMMITS=$1
    local TITLE=$2
    local ARITHMETIC=$3
    local EQUALIZE=$4
    local INPUT=$5

    ./tools/benchmark-compare-commits.sh "${INDEX_COMMITS}" "${ARITHMETIC}" "${EQUALIZE}" "${INPUT}" 2>&1 | tee "compare-commits.${TITLE}.log"
}


build_commits main HEAD
index_commits "${COMMITS[@]}"
sleep 15

echo "[*] Comparing commits for each scenario"
benchmark "${INDEX_COMMITS}" "1k-fixed9"          fixed9    ""          "testdata/ballots/random/rand_hypergeometric.blt"
benchmark "${INDEX_COMMITS}" "1k-lexico-fixed9"   fixed9    ""          "testdata/shuffle_ballots/rand_sorted_lexicographically.blt"
benchmark "${INDEX_COMMITS}" "1k-equalize-fixed9" fixed9    --equalize  "testdata/ballots/random/rand_hypergeometric.blt"
benchmark "${INDEX_COMMITS}" "1k-bigfixed9"       bigfixed9 ""          "testdata/ballots/random/rand_hypergeometric.blt"
benchmark "${INDEX_COMMITS}" "skewed-fixed9"      fixed9    ""          "testdata/ballots/skewed.blt"
benchmark "${INDEX_COMMITS}" "10k-lexico-fixed9"  fixed9    ""          "testdata/shuffle_ballots/rand_10k_sorted_lexicographically.blt"
benchmark "${INDEX_COMMITS}" "100k-lexico-fixed9" fixed9    ""          "testdata/shuffle_ballots/rand_100k_sorted_lexicographically.blt"
