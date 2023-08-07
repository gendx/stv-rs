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

echo "[*] Building at all commits"
./tools/build-all-commits.sh ${COMMITS}
sleep 15

echo "[*] Benchmarking each commit"
ITER=0
for COMMIT in ${COMMITS}
do
    ITER=$(expr ${ITER} + 1)
    printf -v INDEX "%02d" ${ITER}
    echo "[${INDEX}] Benchmarking ${COMMIT}"

    ./tools/benchmark-at.sh ${INDEX} ${COMMIT} 2>&1 | tee ${INDEX}-${COMMIT}.benchmark.log
done
