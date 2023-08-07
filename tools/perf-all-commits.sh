#!/bin/bash

set -eux

BEGIN=main
END=HEAD

echo "[*] Listing commits"
git log ${BEGIN}..${END} --reverse --format=oneline
COMMITS=$(git log ${BEGIN}..${END} --reverse --format=oneline | cut -d' ' -f1)

echo "[*] Building at all commits"
./tools/build-all-commits.sh ${COMMITS}
sleep 15

echo "[*] Perfing each commit"
ITER=0
for COMMIT in ${COMMITS}
do
    ITER=$(expr ${ITER} + 1)
    printf -v INDEX "%02d" ${ITER}
    echo "[${INDEX}] Benchmarking ${COMMIT}"

    ./tools/perf-stat-at.sh ${INDEX} ${COMMIT} 2>&1 | tee ${INDEX}-${COMMIT}.perf.log
done

