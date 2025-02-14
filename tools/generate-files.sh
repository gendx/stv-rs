#!/bin/bash

set -eux

SEED=43

cargo build --release
cargo +nightly build --release --examples

# Help
target/release/stv-rs -h > man/short_help.txt
target/release/stv-rs --help > man/help.txt
target/release/stv-rs meek --help > man/meek_help.txt
target/release/stv-rs plurality --help > man/plurality_help.txt
target/release/stv-rs --version > man/version.txt

# Ballot files
target/release/examples/random_ballots \
  --title "Vegetable contest with many candidates" \
  --ballot-pattern hypergeometric80 \
  --num-ballots 200 \
  --seed $SEED \
  --output testdata/ballots/bigint/rand_hypergeometric_many.blt

target/release/examples/random_ballots \
  --ballot-pattern geometric20 \
  --seed $SEED \
  --output testdata/ballots/crashes/rand_geometric.blt

target/release/examples/random_ballots \
  --ballot-pattern tuples2x10 \
  --seed $SEED \
  --output testdata/ballots/random/rand_2x10.blt

target/release/examples/random_ballots \
  --ballot-pattern tuples5x4 \
  --seed $SEED \
  --output testdata/ballots/random/rand_5x4.blt

target/release/examples/random_ballots \
  --ballot-pattern hypergeometric20 \
  --seed $SEED \
  --output testdata/ballots/random/rand_hypergeometric.blt

target/release/examples/random_ballots \
  --ballot-pattern hypergeometric20 \
  --num-ballots 10000 \
  --seed $SEED \
  --output testdata/ballots/random/rand_hypergeometric_10k.blt

target/release/examples/random_ballots \
  --ballot-pattern hypergeometric20 \
  --num-ballots 100000 \
  --seed $SEED \
  --output testdata/ballots/random/rand_hypergeometric_100k.blt

target/release/examples/random_ballots \
  --ballot-pattern mixed20 \
  --num-ballots 5000 \
  --seed $SEED \
  --output testdata/ballots/random/rand_mixed_5k.blt

# Plurality
for FILE in \
  random/rand_2x10 \
  random/rand_5x4 \
  random/rand_hypergeometric \
  random/rand_hypergeometric_10k \
  random/rand_hypergeometric_100k \
  vegetables
do
  target/release/stv-rs \
    --arithmetic fixed9 --input testdata/ballots/$FILE.blt plurality \
    > testdata/plurality_block_voting/$FILE.fixed9.log
done

# Histogram
for FILE in \
  bigint/ballot_count_overflows \
  bigint/ballot_sum_overflows \
  bigint/rand_hypergeometric_many \
  equal_preference/always_behind \
  equal_preference/equal_preference \
  random/rand_2x10 \
  random/rand_5x4 \
  random/rand_hypergeometric \
  random/rand_hypergeometric_10k \
  random/rand_hypergeometric_100k \
  random/rand_mixed_5k \
  vegetables
do
  target/release/examples/histogram \
    < testdata/ballots/$FILE.blt plurality \
    > testdata/histogram/$FILE.csv
done

# Meek
for FILE in \
  equal_preference/always_behind \
  equal_preference/equal_preference \
  equal_preference/equal_preference_droop \
  equal_preference/equal_preference_equalize \
  negative_surplus/below_quota \
  random/rand_2x10 \
  random/rand_5x4 \
  random/rand_hypergeometric \
  random/rand_hypergeometric_10k \
  random/rand_hypergeometric_100k \
  random/rand_mixed_5k \
  recursive_descent/transfer \
  recursive_descent/transfer_is_blocked \
  ties/tie_break_explicit \
  ties/tie_break_implicit \
  skewed \
  vegetables
do
  target/release/stv-rs \
    --arithmetic fixed9 --input testdata/ballots/$FILE.blt meek \
    > testdata/meek/$FILE.fixed9.log
  target/release/stv-rs \
    --arithmetic fixed9 --input testdata/ballots/$FILE.blt meek --equalize \
    > testdata/meek/$FILE.equalize.fixed9.log
done

# Arithmetic
for FILE in \
  crashes/elect_too_many \
  random/rand_2x10 \
  random/rand_5x4 \
  random/rand_hypergeometric \
  random/rand_mixed_5k \
  vegetables
do
  target/release/stv-rs \
    --arithmetic approx --input testdata/ballots/$FILE.blt meek \
    > testdata/meek/$FILE.approx.log
  target/release/stv-rs \
    --arithmetic approx --input testdata/ballots/$FILE.blt meek --equalize \
    > testdata/meek/$FILE.equalize.approx.log
done

# Surplus tests omitted.
# Numeric tests omitted.

# Shuffle
target/release/examples/shuffle_ballots \
  --strategy product \
  < testdata/ballots/random/rand_hypergeometric.blt \
  > testdata/shuffle_ballots/rand_sorted_by_product.blt

target/release/examples/shuffle_ballots \
  --strategy lexicographic \
  < testdata/ballots/random/rand_hypergeometric.blt \
  > testdata/shuffle_ballots/rand_sorted_lexicographically.blt

target/release/examples/shuffle_ballots \
  --strategy lexico-product \
  < testdata/ballots/random/rand_hypergeometric.blt \
  > testdata/shuffle_ballots/rand_sorted_by_lexico_product.blt

target/release/examples/shuffle_ballots \
  --strategy alphabetic \
  < testdata/ballots/random/rand_mixed_5k.blt \
  > testdata/shuffle_ballots/rand_mixed_5k_sorted_alphabetically.blt

target/release/examples/shuffle_ballots \
  --strategy product \
  < testdata/ballots/random/rand_hypergeometric_10k.blt \
  > testdata/shuffle_ballots/rand_10k_sorted_by_product.blt

target/release/examples/shuffle_ballots \
  --strategy lexicographic \
  < testdata/ballots/random/rand_hypergeometric_10k.blt \
  > testdata/shuffle_ballots/rand_10k_sorted_lexicographically.blt

target/release/examples/shuffle_ballots \
  --strategy lexico-product \
  < testdata/ballots/random/rand_hypergeometric_10k.blt \
  > testdata/shuffle_ballots/rand_10k_sorted_by_lexico_product.blt

# More tests using random ballots
for ARITHMETIC in fixed9 bigfixed9
do
  target/release/stv-rs \
    --arithmetic $ARITHMETIC \
    --input testdata/ballots/bigint/rand_hypergeometric_many.blt \
    meek --equalize \
    > testdata/meek/bigint/rand_hypergeometric_many.equalize.$ARITHMETIC.log \
    2> testdata/meek/bigint/rand_hypergeometric_many.equalize.$ARITHMETIC.err \
    || true
done

target/release/stv-rs \
  --arithmetic fixed9 \
  --input testdata/ballots/crashes/rand_geometric.blt \
  meek --force-positive-surplus \
  > testdata/meek/crashes/rand_geometric.fixed9.log

target/release/stv-rs \
  --arithmetic fixed9 \
  --input testdata/ballots/crashes/rand_geometric.blt \
  meek --force-positive-surplus --equalize \
  > testdata/meek/crashes/rand_geometric.equalize.fixed9.log
