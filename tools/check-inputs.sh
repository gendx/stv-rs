#!/bin/bash

set -eu

if [ ! -f "testdata/ballots/random/rand_hypergeometric_10k.blt" ]; then
    echo "Couldn't find input file: testdata/ballots/random/rand_hypergeometric_10k.blt"
    echo "Please generate it with: './tools/create-large-inputs.sh'"
    exit 42
fi
if [ ! -f "testdata/ballots/random/rand_hypergeometric_100k.blt" ]; then
    echo "Couldn't find input file: testdata/ballots/random/rand_hypergeometric_100k.blt"
    echo "Please generate it with: './tools/create-large-inputs.sh'"
    exit 42
fi

if [ ! -f "testdata/shuffle_ballots/rand_10k_sorted_lexicographically.blt" ]; then
    echo "Couldn't find input file: testdata/shuffle_ballots/rand_10k_sorted_lexicographically.blt"
    echo "Please generate it with: './tools/create-large-inputs.sh'"
    exit 42
fi
if [ ! -f "testdata/shuffle_ballots/rand_100k_sorted_lexicographically.blt" ]; then
    echo "Couldn't find input file: testdata/shuffle_ballots/rand_100k_sorted_lexicographically.blt"
    echo "Please generate it with: './tools/create-large-inputs.sh'"
    exit 42
fi
