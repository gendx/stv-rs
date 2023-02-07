// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Module to count votes, based on input ballots and the current keep factor
//! values.

use crate::arithmetic::{Integer, Rational};
use crate::types::{Ballot, Election};
use log::{debug, trace, warn};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

/// Result of a vote count.
pub struct VoteCount<I, R> {
    /// Sum of votes for each candidate.
    pub sum: Vec<R>,
    /// Exhausted voting power.
    pub exhausted: R,
    _phantom: PhantomData<I>,
}

struct VoteAccumulator<I, R> {
    /// Sum of votes for each candidate.
    sum: Vec<R>,
    /// Exhausted voting power.
    exhausted: R,
    /// Number of function calls performed.
    fn_calls: usize,
    _phantom: PhantomData<I>,
}

impl<I, R> VoteAccumulator<I, R>
where
    I: Integer,
    for<'a> &'a I: Add<&'a I, Output = I>,
    for<'a> &'a I: Sub<&'a I, Output = I>,
    for<'a> &'a I: Mul<&'a I, Output = I>,
    R: Rational<I>,
    for<'a> &'a R: Add<&'a R, Output = R>,
    for<'a> &'a R: Sub<&'a R, Output = R>,
    for<'a> &'a R: Mul<&'a R, Output = R>,
    for<'a> &'a R: Mul<&'a I, Output = R>,
    for<'a> &'a R: Div<&'a R, Output = R>,
    for<'a> &'a R: Div<&'a I, Output = R>,
{
    fn new(num_candidates: usize) -> Self {
        Self {
            sum: vec![R::zero(); num_candidates],
            exhausted: R::zero(),
            fn_calls: 0,
            _phantom: PhantomData,
        }
    }

    fn reduce(self, other: Self) -> Self {
        Self {
            sum: std::iter::zip(self.sum, other.sum)
                .map(|(a, b)| a + b)
                .collect(),
            exhausted: self.exhausted + other.exhausted,
            fn_calls: self.fn_calls + other.fn_calls,
            _phantom: PhantomData,
        }
    }

    fn into_vote_count(self) -> VoteCount<I, R> {
        VoteCount {
            sum: self.sum,
            exhausted: self.exhausted,
            _phantom: PhantomData,
        }
    }
}

impl<I, R> VoteCount<I, R>
where
    I: Integer + Send + Sync,
    for<'a> &'a I: Add<&'a I, Output = I>,
    for<'a> &'a I: Sub<&'a I, Output = I>,
    for<'a> &'a I: Mul<&'a I, Output = I>,
    R: Rational<I> + Send + Sync,
    for<'a> &'a R: Add<&'a R, Output = R>,
    for<'a> &'a R: Sub<&'a R, Output = R>,
    for<'a> &'a R: Mul<&'a R, Output = R>,
    for<'a> &'a R: Mul<&'a I, Output = R>,
    for<'a> &'a R: Div<&'a R, Output = R>,
    for<'a> &'a R: Div<&'a I, Output = R>,
{
    /// Prints statistics about this vote count to stdout.
    pub fn print_stats(&self, threshold: &R, surplus: &R) {
        println!("\tQuota: {threshold}");
        let total_votes = self.sum.iter().sum::<R>();
        println!("\tVotes: {total_votes}");
        println!("\tResidual: {}", self.exhausted);
        println!("\tTotal: {}", total_votes + &self.exhausted);
        println!("\tSurplus: {surplus}");
    }

    /// Counts the votes, based on the given keep factors.
    pub fn count_votes(election: &Election, keep_factors: &[R], parallel: bool) -> Self {
        let vote_accumulator = if parallel {
            Self::accumulate_votes_rayon(election, keep_factors)
        } else {
            Self::accumulate_votes_serial(election, keep_factors)
        };

        trace!("Finished counting ballots:");
        for (i, x) in vote_accumulator.sum.iter().enumerate() {
            trace!("  Sum[{i}] = {x} ~ {}", x.to_f64());
        }

        vote_accumulator.into_vote_count()
    }

    /// Serial implementation of vote counting, using only one CPU core.
    fn accumulate_votes_serial(election: &Election, keep_factors: &[R]) -> VoteAccumulator<I, R> {
        let mut vote_accumulator = VoteAccumulator::new(election.num_candidates);
        for (i, ballot) in election.ballots.iter().enumerate() {
            Self::process_ballot(&mut vote_accumulator, keep_factors, i, ballot);
        }
        vote_accumulator
    }

    /// Parallel implementation of vote counting, leveraging all CPU cores to
    /// speed up the computation.
    fn accumulate_votes_rayon(election: &Election, keep_factors: &[R]) -> VoteAccumulator<I, R> {
        election
            .ballots
            .par_iter()
            .enumerate()
            .map(|(i, ballot)| {
                let mut vote_accumulator = VoteAccumulator::new(election.num_candidates);
                Self::process_ballot(&mut vote_accumulator, keep_factors, i, ballot);
                vote_accumulator
            })
            .reduce(
                || VoteAccumulator::new(election.num_candidates),
                |a, b| a.reduce(b),
            )
    }

    /// Processes a ballot and adds its votes to the accumulator.
    fn process_ballot(
        vote_accumulator: &mut VoteAccumulator<I, R>,
        keep_factors: &[R],
        i: usize,
        ballot: &Ballot,
    ) {
        trace!("Processing ballot {i} = {:?}", ballot.order);

        let mut unused_power = R::from_usize(ballot.count);
        let counter = BallotCounter {
            ballot,
            sum: &mut vote_accumulator.sum,
            unused_power: &mut unused_power,
            keep_factors,
            fn_calls: &mut vote_accumulator.fn_calls,
            _phantom: PhantomData,
        };
        counter.process_ballot_rec();

        if !unused_power.is_zero() {
            let pwr = &unused_power / &I::from_usize(ballot.count);
            trace!("  Exhausted voting_power = {pwr} ~ {}", pwr.to_f64());
        } else {
            trace!("  Exhausted voting_power is zero :)");
        }

        vote_accumulator.exhausted += unused_power;
    }

    /// Computes the new threshold, based on the election parameters and the
    /// exhausted votes after this count.
    pub fn threshold(&self, election: &Election) -> R {
        Self::threshold_exhausted(election, &self.exhausted)
    }

    /// Computes the current surplus, based on the votes, the threshold and the
    /// currently elected candidates. This is the sum of the differences between
    /// the received vote count and the required threshold, across all
    /// elected candidates.
    ///
    /// Note: Normally we'd only add positive surpluses here, but sometimes an
    /// already elected candidate goes below the threshold. Instead, we just
    /// sum the difference for the elected candidates.
    pub fn surplus(&self, threshold: &R, elected: &[usize]) -> R {
        elected
            .iter()
            .map(|&i| {
                let x = &self.sum[i];
                if x < threshold {
                    warn!(
                        "Candidate #{i} was elected but received fewer votes than the threshold."
                    );
                }
                x - threshold
            })
            .sum()
    }

    /// Computes the new threshold, based on the given election parameters and
    /// exhausted votes.
    ///
    /// This is the ratio between the effectively used voting power (i.e.
    /// subtracted from any exhausted voting power) and the number of elected
    /// seats plus one.
    pub fn threshold_exhausted(election: &Election, exhausted: &R) -> R {
        let numerator = R::from_usize(election.num_ballots) - exhausted;
        let denominator = R::from_usize(election.num_seats + 1);
        let result = &numerator / &denominator + R::epsilon();
        debug!(
            "threshold_exhausted = {numerator} / {denominator} + {} ~ {} / {} + {}",
            R::epsilon(),
            numerator.to_f64(),
            denominator.to_f64(),
            R::epsilon().to_f64(),
        );
        debug!("\t= {result} ~ {}", result.to_f64());
        result
    }
}

struct BallotCounter<'a, I, R> {
    /// Ballot to count.
    ballot: &'a Ballot,
    /// Candidates' attributed votes.
    sum: &'a mut [R],
    /// Voting power that was not used (due to rounding and/or voting power not
    /// kept by any candidate in the ballot).
    unused_power: &'a mut R,
    /// Candidates' keep factors.
    keep_factors: &'a [R],
    /// Number of functions called to count this ballot (due to recursion).
    fn_calls: &'a mut usize,
    _phantom: PhantomData<I>,
}

impl<I, R> BallotCounter<'_, I, R>
where
    I: Integer,
    for<'a> &'a I: Add<&'a I, Output = I>,
    for<'a> &'a I: Sub<&'a I, Output = I>,
    for<'a> &'a I: Mul<&'a I, Output = I>,
    R: Rational<I>,
    for<'a> &'a R: Add<&'a R, Output = R>,
    for<'a> &'a R: Sub<&'a R, Output = R>,
    for<'a> &'a R: Mul<&'a R, Output = R>,
    for<'a> &'a R: Mul<&'a I, Output = R>,
    for<'a> &'a R: Div<&'a R, Output = R>,
    for<'a> &'a R: Div<&'a I, Output = R>,
{
    /// Processes a ballot, using a recursive method (consistent with Droop.py)
    /// to count ballots that contain candidates ranked equally.
    fn process_ballot_rec(mut self) {
        let voting_power = R::one();
        let ballot_count = I::from_usize(self.ballot.count);
        if self.ballot.order.iter().all(|ranking| ranking.len() == 1) {
            self.process_ballot_rec_notie(voting_power, &ballot_count);
        } else {
            self.process_ballot_rec_impl(voting_power, &ballot_count, 0);
        }
    }

    /// Processes a ballot which contains a strict order of candidates (no tie),
    /// in a manner consistent with Droop.py.
    ///
    /// This is a straightforward application of
    /// [Meek's rules](https://en.wikipedia.org/wiki/Counting_single_transferable_votes#Meek):
    /// each candidate in the ballot retains its own keep factor share of the
    /// voting power, and passes the rest to the next candidate, until no voting
    /// power is left (i.e. a candidate has a keep factor of 1), or the whole
    /// ballot was used.
    fn process_ballot_rec_notie(&mut self, mut voting_power: R, ballot_count: &I) {
        *self.fn_calls += 1;

        for ranking in &self.ballot.order {
            // Only one candidate at this level. Easy case.
            let candidate = ranking[0];
            let (used_power, remaining_power) =
                self.split_power_one_candidate(&voting_power, candidate);
            self.increment_candidate(candidate, used_power, ballot_count);
            voting_power = remaining_power;
            if voting_power.is_zero() {
                break;
            }
        }
    }

    /// Processes a ballot which contains at least a tie, in a manner consistent
    /// with [Droop.py](https://github.com/jklundell/droop). There are two
    /// remarks to make about this method.
    ///
    /// First, if a ranking among the ballot only contains defeated candidates,
    /// then the remaining power is not distributed any further, which is
    /// contrary to Meek's method (a defeated candidate should pass all of its
    /// voting power to the next candidate).
    ///
    /// For example, with the ballot `a b c=d`, if `a` is defeated then no
    /// voting power will be distributed to `b`, `c` nor `d`. On the contrary,
    /// the ballot `a b c` would get voting power distributed to `b` if `a`
    /// was defeated (via [`Self::process_ballot_rec_notie()`]).
    ///
    /// Second, if multiple candidates are ranked equally, the voting power is
    /// first split evenly between them. But if any of those has a keep factor
    /// lower than 1, then the remaining voting power is not contributed back to
    /// any of these tied candidates - it is instead distributed to candidates
    /// ranked afterwards.
    ///
    /// This means that with the ballot `a=b c`, if `a` is elected with a keep
    /// factor k, then `a` will receive `0.5 * k`, `b` will receive `0.5`, and
    /// `c` will receive `0.5 * (1 - k)`. This is contrary to what one would
    /// expect: `c` should only receive voting power from this ballot if
    /// both `a` and `b` are already elected.
    fn process_ballot_rec_impl(&mut self, mut voting_power: R, ballot_count: &I, i: usize) {
        *self.fn_calls += 1;
        if i == self.ballot.order.len() {
            return;
        }

        // Fetch the i-th ranking in the ballot, which may contain multiple tied
        // candidates.
        let ranking = &self.ballot.order[i];

        // TODO: skip defeated candidates according to their status.
        // Number of eligible candidates (with a positive keep factor) in the current
        // ranking.
        let filtered_ranking_len = ranking
            .iter()
            .filter(|&&candidate| !self.keep_factors[candidate].is_zero())
            .count();
        // If all the candidates are defeated in the current ranking, the original
        // implementation (from Droop.py) doesn't distribute votes any further!
        if filtered_ranking_len == 0 {
            return;
        }

        // Multiple candidates ranked equally. The original implementation (from
        // Droop.py) starts by splitting the voting power evenly to all these
        // candidates. If any of those has a non-one keep factor (i.e. is already
        // elected), then the remaining power (for that candidate) is distributed via
        // recursion to the next ranking in the ballot, rather than being distributed
        // back to any other candidate in the same ranking.
        voting_power /= &I::from_usize(filtered_ranking_len);
        for &candidate in ranking
            .iter()
            .filter(|&&candidate| !self.keep_factors[candidate].is_zero())
        {
            let (used_power, remaining_power) =
                self.split_power_one_candidate(&voting_power, candidate);
            self.increment_candidate(candidate, used_power, ballot_count);
            if !remaining_power.is_zero() {
                self.process_ballot_rec_impl(remaining_power, ballot_count, i + 1)
            }
        }
    }

    /// Increments the votes attributed to a candidate.
    ///
    /// Note: we separate the voting power and the number of occurrences of each
    /// ballot, and multiply by the ballot count only at the end, so that in
    /// order to ensure fairness a repeated ballot is really counted as the sum
    /// of `ballot_count` identical ballots.
    fn increment_candidate(&mut self, candidate: usize, used_power: R, ballot_count: &I) {
        let scaled_power = &used_power * ballot_count;
        self.sum[candidate] += &scaled_power;
        *self.unused_power -= &scaled_power;
    }

    /// Splits the input voting power for a candidate into the used power and
    /// the remaining power, according to the candidate's keep factor.
    fn split_power_one_candidate(&self, voting_power: &R, candidate: usize) -> (R, R) {
        let keep_factor = &self.keep_factors[candidate];
        // The candidate uses only a fraction k of the voting power, equal to its keep
        // factor.
        let used_power = voting_power * keep_factor;
        // The remaining voting power cannot be larger than a `(1 - k)` fraction of the
        // original voting power. Due to the potential rounding when computing the
        // multiplication `power * k`, the remaining power needs to be explicilty
        // computed as `power * (1 - k)`, rather than `power - used_power`.
        let remaining_power = voting_power * &(R::one() - keep_factor);

        trace!(
            "  Split[{candidate}]: used = {used_power} ~ {} | remaining = {remaining_power} ~ {}",
            used_power.to_f64(),
            remaining_power.to_f64(),
        );

        let total = &used_power + &remaining_power;
        // Note: This assertion can fail when using f64.
        if &total > voting_power {
            trace!(
                "used + remaining > voting | {} + {} = {} > {}",
                used_power.to_f64(),
                remaining_power.to_f64(),
                total.to_f64(),
                voting_power.to_f64()
            );
        }

        (used_power, remaining_power)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::arithmetic::{ApproxRational, BigFixedDecimal9, FixedDecimal9};
    #[cfg(feature = "benchmarks")]
    use ::test::Bencher;
    use num::rational::Ratio;
    use num::{BigInt, BigRational};
    use std::fmt::Display;
    #[cfg(feature = "benchmarks")]
    use std::hint::black_box;

    macro_rules! numeric_tests {
        ( $typei:ty, $typer:ty, $($case:ident,)+ ) => {
            $(
            #[test]
            fn $case() {
                $crate::vote_count::test::NumericTests::<$typei, $typer>::$case();
            }
            )+
        };
    }

    #[cfg(feature = "benchmarks")]
    macro_rules! numeric_benches {
        ( $typei:ty, $typer:ty, $($case:ident,)+ ) => {
            $(
            #[bench]
            fn $case(b: &mut ::test::Bencher) {
                $crate::vote_count::test::NumericTests::<$typei, $typer>::$case(b);
            }
            )+
        };
    }

    macro_rules! all_numeric_tests {
        ( $mod:ident, $typei:ty, $typer:ty ) => {
            mod $mod {
                use super::*;

                numeric_tests!(
                    $typei,
                    $typer,
                    test_process_ballot_rec_first,
                    test_process_ballot_rec_chain,
                    test_process_ballot_rec_defeated,
                    test_process_ballot_rec_tie_first,
                    test_process_ballot_rec_tie_chain,
                    test_process_ballot_rec_tie_defeated,
                    test_process_ballot_rec_ties,
                );

                #[cfg(feature = "benchmarks")]
                numeric_benches!(
                    $typei,
                    $typer,
                    bench_process_ballot_rec_chain,
                    bench_process_ballot_rec_pairs,
                    bench_process_ballot_rec_tens,
                );
            }
        };
    }

    all_numeric_tests!(ratio_i64, i64, Ratio<i64>);
    all_numeric_tests!(exact, BigInt, BigRational);
    all_numeric_tests!(approx_rational, BigInt, ApproxRational);
    all_numeric_tests!(fixed9, i64, FixedDecimal9);
    all_numeric_tests!(big_fixed9, BigInt, BigFixedDecimal9);

    pub struct NumericTests<I, R> {
        _phantomi: PhantomData<I>,
        _phantomr: PhantomData<R>,
    }

    impl<I, R> NumericTests<I, R>
    where
        I: Integer + Display,
        for<'a> &'a I: Add<&'a I, Output = I>,
        for<'a> &'a I: Sub<&'a I, Output = I>,
        for<'a> &'a I: Mul<&'a I, Output = I>,
        R: Rational<I>,
        for<'a> &'a R: Add<&'a R, Output = R>,
        for<'a> &'a R: Sub<&'a R, Output = R>,
        for<'a> &'a R: Mul<&'a R, Output = R>,
        for<'a> &'a R: Mul<&'a I, Output = R>,
        for<'a> &'a R: Div<&'a R, Output = R>,
        for<'a> &'a R: Div<&'a I, Output = R>,
    {
        fn process_ballot_rec(
            ballot: impl std::borrow::Borrow<Ballot>,
            keep_factors: &[R],
        ) -> (Vec<R>, R, usize) {
            let num_candidates = keep_factors.len();
            let mut sum = vec![R::zero(); num_candidates];
            let mut unused_power = R::one();
            let mut fn_calls = 0;

            let counter = BallotCounter {
                ballot: ballot.borrow(),
                sum: &mut sum,
                unused_power: &mut unused_power,
                keep_factors,
                fn_calls: &mut fn_calls,
                _phantom: PhantomData,
            };
            counter.process_ballot_rec();

            (sum, unused_power, fn_calls)
        }

        // First candidate takes it all.
        fn test_process_ballot_rec_first() {
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(
                Ballot {
                    count: 1,
                    order: vec![vec![0], vec![1], vec![2]],
                },
                /* keep_factors = */ &[R::one(), R::one(), R::one()],
            );

            assert_eq!(sum, vec![R::one(), R::zero(), R::zero()]);
            assert_eq!(unused_power, R::zero());
            assert_eq!(fn_calls, 1);
        }

        // Chain of keep factors.
        fn test_process_ballot_rec_chain() {
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(
                Ballot {
                    count: 1,
                    order: vec![vec![0], vec![1], vec![2]],
                },
                /* keep_factors = */ &[R::ratio(1, 2), R::ratio(2, 3), R::ratio(3, 4)],
            );

            assert_eq!(sum, vec![R::ratio(1, 2), R::ratio(1, 3), R::ratio(1, 8)]);
            // 1/24.
            assert_eq!(unused_power, R::one() - R::ratio(23, 24));
            assert_eq!(fn_calls, 1);
        }

        // Defeated candidate (keep factor is zero).
        fn test_process_ballot_rec_defeated() {
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(
                Ballot {
                    count: 1,
                    order: vec![vec![0], vec![1], vec![2]],
                },
                /* keep_factors = */ &[R::zero(), R::ratio(2, 3), R::ratio(3, 4)],
            );

            assert_eq!(sum, vec![R::zero(), R::ratio(2, 3), R::ratio(1, 4)]);
            // 1/12.
            assert_eq!(unused_power, R::one() - R::ratio(11, 12));
            assert_eq!(fn_calls, 1);
        }

        // Tie to start the ballot.
        fn test_process_ballot_rec_tie_first() {
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(
                Ballot {
                    count: 1,
                    order: vec![vec![0, 1], vec![2]],
                },
                /* keep_factors = */ &[R::one(), R::one(), R::one()],
            );

            assert_eq!(sum, vec![R::ratio(1, 2), R::ratio(1, 2), R::zero()]);
            assert_eq!(unused_power, R::zero());
            assert_eq!(fn_calls, 1);
        }

        // Tie with chaining.
        fn test_process_ballot_rec_tie_chain() {
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(
                Ballot {
                    count: 1,
                    order: vec![vec![0, 1], vec![2]],
                },
                /* keep_factors = */ &[R::ratio(1, 2), R::ratio(2, 3), R::ratio(3, 4)],
            );

            // The last candidate gets 0.5 * (1/2 + 1/3) * 3/4 = 5/6 * 3/8 = 5/16
            assert_eq!(sum, vec![R::ratio(1, 4), R::ratio(1, 3), R::ratio(5, 16)]);
            // 5/48.
            assert_eq!(unused_power, R::one() - R::ratio(43, 48));
            assert_eq!(fn_calls, 5);
        }

        // Tie with defeated candidate.
        fn test_process_ballot_rec_tie_defeated() {
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(
                Ballot {
                    count: 1,
                    order: vec![vec![0], vec![1, 2]],
                },
                /* keep_factors = */ &[R::zero(), R::ratio(2, 3), R::ratio(3, 4)],
            );

            // No candidate gets anything (Droop.py's behavior)!
            assert_eq!(sum, vec![R::zero(); 3]);
            assert_eq!(unused_power, R::one());
            assert_eq!(fn_calls, 1);
        }

        // Chain of multiple ties.
        fn test_process_ballot_rec_ties() {
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(
                Ballot {
                    count: 1,
                    order: vec![vec![0, 1], vec![2, 3]],
                },
                /* keep_factors = */
                &[
                    R::ratio(1, 2),
                    R::ratio(2, 3),
                    R::ratio(3, 4),
                    R::ratio(4, 5),
                ],
            );

            // Candidates ranked second share 0.5 * (1/2 + 1/3) = 5/12.
            assert_eq!(
                sum,
                vec![
                    R::ratio(1, 4),
                    R::ratio(1, 3),
                    // 0.5 * 5/12 * 3/4 = 5/32
                    R::ratio(5, 24) * R::ratio(3, 4),
                    // 0.5 * 5/12 * 4/5 = 1/6
                    R::ratio(1, 6),
                ]
            );
            // Remaining voting power is 0.5 * (1/4 + 1/5) * 5/12 = 9/40 * 5/12 = 3/32.
            assert_eq!(
                unused_power,
                R::one()
                    - (R::ratio(1, 4)
                        + R::ratio(1, 3)
                        + R::ratio(5, 24) * R::ratio(3, 4)
                        + R::ratio(1, 6))
            );
            assert_eq!(fn_calls, 7);
        }

        #[cfg(feature = "benchmarks")]
        fn bench_process_ballot_rec_chain(bencher: &mut Bencher) {
            let ballot = Ballot {
                count: 1,
                order: (0..10).map(|i| vec![i]).collect(),
            };
            let keep_factors: Vec<R> = (1..=10).map(|i| R::ratio(1, i)).collect();
            bencher.iter(|| Self::process_ballot_rec(black_box(&ballot), black_box(&keep_factors)))
        }

        #[cfg(feature = "benchmarks")]
        fn bench_process_ballot_rec_pairs(bencher: &mut Bencher) {
            let ballot = Ballot {
                count: 1,
                order: (0..10)
                    .collect::<Vec<_>>()
                    .chunks(2)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
            };
            let keep_factors: Vec<R> = (1..=10).map(|i| R::ratio(1, i)).collect();
            bencher.iter(|| Self::process_ballot_rec(black_box(&ballot), black_box(&keep_factors)))
        }

        #[cfg(feature = "benchmarks")]
        fn bench_process_ballot_rec_tens(bencher: &mut Bencher) {
            let ballot = Ballot {
                count: 1,
                order: (0..30)
                    .collect::<Vec<_>>()
                    .chunks(10)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
            };
            let keep_factors: Vec<R> = (1..=30).map(|i| R::ratio(1, i)).collect();
            bencher.iter(|| Self::process_ballot_rec(black_box(&ballot), black_box(&keep_factors)))
        }
    }
}
