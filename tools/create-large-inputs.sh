#!/bin/bash

set -eux

if [ ! -f "testdata/ballots/random/rand_hypergeometric_10k.blt" ]; then
    echo "Generating file: testdata/ballots/random/rand_hypergeometric_10k.blt"
    cargo +nightly run --release --example random_ballots -- --ballot-pattern hypergeometric20 --num-ballots 10000 --seed 43 --output testdata/ballots/random/rand_hypergeometric_10k.blt
fi
if [ ! -f "testdata/ballots/random/rand_hypergeometric_100k.blt" ]; then
    echo "Generating file: testdata/ballots/random/rand_hypergeometric_100k.blt"
    cargo +nightly run --release --example random_ballots -- --ballot-pattern hypergeometric20 --num-ballots 100000 --seed 43 --output testdata/ballots/random/rand_hypergeometric_100k.blt
fi

if [ ! -f "testdata/shuffle_ballots/rand_10k_sorted_lexicographically.blt" ]; then
    echo "Generating file: testdata/shuffle_ballots/rand_10k_sorted_lexicographically.blt"
    cargo +nightly run --release --example shuffle_ballots -- --strategy lexicographic < testdata/ballots/random/rand_hypergeometric_10k.blt > testdata/shuffle_ballots/rand_10k_sorted_lexicographically.blt
fi
if [ ! -f "testdata/shuffle_ballots/rand_100k_sorted_lexicographically.blt" ]; then
    echo "Generating file: testdata/shuffle_ballots/rand_100k_sorted_lexicographically.blt"
    cargo +nightly run --release --example shuffle_ballots -- --strategy lexicographic < testdata/ballots/random/rand_hypergeometric_100k.blt > testdata/shuffle_ballots/rand_100k_sorted_lexicographically.blt
fi

