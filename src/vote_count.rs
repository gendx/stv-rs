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

use crate::arithmetic::{Integer, IntegerRef, Rational, RationalRef};
use crate::cli::Parallel;
use crate::parallelism::{RangeStrategy, ThreadAccumulator, ThreadPool};
use crate::types::{Ballot, Election};
use log::Level::{Trace, Warn};
use log::{debug, log_enabled, trace, warn};
use rayon::prelude::*;
use std::io;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock, RwLockReadGuard};
use std::thread::Scope;

/// Result of a vote count.
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct VoteCount<I, R> {
    /// Sum of votes for each candidate.
    pub sum: Vec<R>,
    /// Exhausted voting power.
    pub exhausted: R,
    _phantom: PhantomData<I>,
}

#[cfg(test)]
impl<I, R> VoteCount<I, R>
where
    R: Clone,
{
    pub(crate) fn new(sum: impl Into<Vec<R>>, exhausted: R) -> Self {
        VoteCount {
            sum: sum.into(),
            exhausted,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct VoteAccumulator<I, R> {
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
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I>,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    pub(crate) fn new(num_candidates: usize) -> Self {
        Self {
            sum: vec![R::zero(); num_candidates],
            exhausted: R::zero(),
            fn_calls: 0,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn reduce(self, other: Self) -> Self {
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

/// A thread pool tied to a scope, that can perform vote counting rounds.
pub struct VoteCountingThreadPool<'scope, I, R> {
    /// Inner thread pool.
    pool: ThreadPool<'scope, VoteAccumulator<I, R>>,
    /// Storage for the keep factors, used as input of the current round by the
    /// worker threads.
    keep_factors: Arc<RwLock<Vec<R>>>,
}

impl<'scope, I, R> VoteCountingThreadPool<'scope, I, R>
where
    I: Integer + Send + Sync + 'scope,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I> + Send + Sync + 'scope,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    /// Creates a new pool tied to the given scope, with the given number of
    /// threads and references to the necessary election inputs.
    pub fn new<'env>(
        thread_scope: &'scope Scope<'scope, 'env>,
        num_threads: NonZeroUsize,
        range_strategy: RangeStrategy,
        election: &'env Election,
        pascal: Option<&'env [Vec<I>]>,
    ) -> Self {
        let keep_factors = Arc::new(RwLock::new(Vec::new()));
        Self {
            pool: ThreadPool::new(
                thread_scope,
                num_threads,
                range_strategy,
                &election.ballots,
                || ThreadVoteCounter {
                    num_candidates: election.num_candidates,
                    pascal,
                    keep_factors: keep_factors.clone(),
                },
            ),
            keep_factors,
        }
    }

    /// Accumulates votes from the election ballots based on the given keep
    /// factors.
    pub fn accumulate_votes(&self, keep_factors: &[R]) -> VoteAccumulator<I, R> {
        {
            let mut keep_factors_guard = self.keep_factors.write().unwrap();
            keep_factors_guard.clear();
            keep_factors_guard.extend_from_slice(keep_factors);
        }

        self.pool
            .process_inputs()
            .reduce(|a, b| a.reduce(b))
            .unwrap()
    }
}

/// Helper state to accumulate votes in a worker thread.
struct ThreadVoteCounter<'env, I, R> {
    /// Number of candidates in the election.
    num_candidates: usize,
    /// Pre-computed Pascal triangle.
    pascal: Option<&'env [Vec<I>]>,
    /// Keep factors used in the current round.
    keep_factors: Arc<RwLock<Vec<R>>>,
}

impl<I, R> ThreadAccumulator<Ballot, VoteAccumulator<I, R>> for ThreadVoteCounter<'_, I, R>
where
    I: Integer,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I>,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    type Accumulator<'a>
        = (VoteAccumulator<I, R>, RwLockReadGuard<'a, Vec<R>>)
    where
        Self: 'a,
        I: 'a,
        R: 'a;

    fn init(&self) -> Self::Accumulator<'_> {
        (
            VoteAccumulator::new(self.num_candidates),
            self.keep_factors.read().unwrap(),
        )
    }

    fn process_item<'a>(
        &'a self,
        accumulator: &mut Self::Accumulator<'a>,
        index: usize,
        ballot: &Ballot,
    ) {
        let (vote_accumulator, keep_factors) = accumulator;
        VoteCount::<I, R>::process_ballot(
            vote_accumulator,
            keep_factors,
            self.pascal,
            index,
            ballot,
        );
    }

    fn finalize<'a>(&'a self, accumulator: Self::Accumulator<'a>) -> VoteAccumulator<I, R> {
        accumulator.0
    }
}

impl<I, R> VoteCount<I, R>
where
    I: Integer + Send + Sync,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I> + Send + Sync,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    /// Counts the votes, based on the given keep factors.
    pub fn count_votes(
        election: &Election,
        keep_factors: &[R],
        parallel: Parallel,
        thread_pool: Option<&VoteCountingThreadPool<'_, I, R>>,
        pascal: Option<&[Vec<I>]>,
    ) -> Self {
        let vote_accumulator = match parallel {
            Parallel::No => Self::accumulate_votes_serial(election, keep_factors, pascal),
            Parallel::Rayon => Self::accumulate_votes_rayon(election, keep_factors, pascal),
            Parallel::Custom => thread_pool.unwrap().accumulate_votes(keep_factors),
        };

        if log_enabled!(Trace) {
            trace!("Finished counting ballots:");
            for (i, x) in vote_accumulator.sum.iter().enumerate() {
                trace!("  Sum[{i}] = {x} ~ {}", x.to_f64());
            }
        }

        let equalize = pascal.is_some();
        if !equalize {
            debug!(
                "Finished counting votes ({} recursive calls)",
                vote_accumulator.fn_calls
            );
        }

        vote_accumulator.into_vote_count()
    }

    /// Parallel implementation of vote counting, leveraging all CPU cores to
    /// speed up the computation.
    fn accumulate_votes_rayon(
        election: &Election,
        keep_factors: &[R],
        pascal: Option<&[Vec<I>]>,
    ) -> VoteAccumulator<I, R> {
        election
            .ballots
            .par_iter()
            .enumerate()
            .fold_with(
                VoteAccumulator::new(election.num_candidates),
                |mut vote_accumulator, (i, ballot)| {
                    Self::process_ballot(&mut vote_accumulator, keep_factors, pascal, i, ballot);
                    vote_accumulator
                },
            )
            .reduce(
                || VoteAccumulator::new(election.num_candidates),
                |a, b| a.reduce(b),
            )
    }
}

impl<I, R> VoteCount<I, R>
where
    I: Integer,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I>,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    /// Writes statistics about this vote count to the given output.
    pub fn write_stats(
        &self,
        out: &mut impl io::Write,
        threshold: &R,
        surplus: &R,
    ) -> io::Result<()> {
        writeln!(out, "\tQuota: {threshold}")?;
        let total_votes = self.sum.iter().sum::<R>();
        writeln!(out, "\tVotes: {total_votes}")?;
        writeln!(out, "\tResidual: {}", self.exhausted)?;
        writeln!(out, "\tTotal: {}", total_votes + &self.exhausted)?;
        writeln!(out, "\tSurplus: {surplus}")?;
        Ok(())
    }

    /// Serial implementation of vote counting, using only one CPU core.
    fn accumulate_votes_serial(
        election: &Election,
        keep_factors: &[R],
        pascal: Option<&[Vec<I>]>,
    ) -> VoteAccumulator<I, R> {
        let mut vote_accumulator = VoteAccumulator::new(election.num_candidates);
        for (i, ballot) in election.ballots.iter().enumerate() {
            Self::process_ballot(&mut vote_accumulator, keep_factors, pascal, i, ballot);
        }
        vote_accumulator
    }

    /// Processes a ballot and adds its votes to the accumulator.
    pub(crate) fn process_ballot(
        vote_accumulator: &mut VoteAccumulator<I, R>,
        keep_factors: &[R],
        pascal: Option<&[Vec<I>]>,
        i: usize,
        ballot: &Ballot,
    ) {
        trace!("Processing ballot {i} = {:?}", ballot);

        let mut unused_power = R::from_usize(ballot.count());
        let counter = BallotCounter {
            ballot,
            sum: &mut vote_accumulator.sum,
            unused_power: &mut unused_power,
            keep_factors,
            fn_calls: &mut vote_accumulator.fn_calls,
            pascal,
            _phantom: PhantomData,
        };

        let equalize = pascal.is_some();
        if equalize {
            counter.process_ballot_equalize()
        } else {
            counter.process_ballot_rec()
        };

        if log_enabled!(Trace) {
            if !unused_power.is_zero() {
                let pwr = &unused_power / &I::from_usize(ballot.count());
                trace!("  Exhausted voting_power = {pwr} ~ {}", pwr.to_f64());
            } else {
                trace!("  Exhausted voting_power is zero :)");
            }
        }

        vote_accumulator.exhausted += unused_power;
    }

    /// Computes the new threshold, based on the election parameters and the
    /// exhausted votes after this count.
    pub fn threshold(&self, election: &Election) -> R {
        Self::threshold_exhausted(election, &self.exhausted)
    }

    /// Computes the current surplus, based on the votes, the threshold and the
    /// currently elected candidates, in a manner compatible with Droop.py. This
    /// is the sum of the differences between the received vote count and
    /// the required threshold, across all elected candidates.
    ///
    /// Note: Normally we'd only add positive surpluses here, but sometimes an
    /// already elected candidate goes below the threshold. In this case,
    /// Droop.py just sums the difference for the elected candidates, which
    /// can lead to crashes. See the [`Self::surplus_positive()`] function which
    /// fixes this behavior.
    pub fn surplus_droop(&self, threshold: &R, elected: &[usize]) -> R {
        elected
            .iter()
            .map(|&i| {
                let result = &self.sum[i] - threshold;
                if log_enabled!(Warn) && result < R::zero() {
                    warn!(
                        "Candidate #{i} was elected but received fewer votes than the threshold."
                    );
                }
                result
            })
            .sum()
    }

    /// Computes the current surplus, based on the votes, the threshold and the
    /// currently elected candidates. This is the sum of the differences between
    /// the received vote count and the required threshold, across all
    /// elected candidates.
    ///
    /// Note: Normally we'd only add positive surpluses here, but sometimes an
    /// already elected candidate goes below the threshold. In this case, this
    /// function counts a positive surplus for this candidate. See the
    /// [`Self::surplus_droop()`] function for Droop.py's behavior.
    pub fn surplus_positive(&self, threshold: &R, elected: &[usize]) -> R {
        elected
            .iter()
            .map(|&i| {
                let x = &self.sum[i];
                if x < threshold {
                    warn!(
                        "Candidate #{i} was elected but received fewer votes than the threshold."
                    );
                    R::zero()
                } else {
                    x - threshold
                }
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
        let denominator = I::from_usize(election.num_seats + 1);
        let result = &numerator / &denominator + R::epsilon();
        debug!(
            "threshold_exhausted = {numerator} / {denominator:?} + {} ~ {} / {:?} + {}",
            R::epsilon(),
            numerator.to_f64(),
            R::from_int(denominator.clone()).to_f64(),
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
    /// Pre-computed Pascal triangle. Set only if the "equalized counting" is
    /// enabled.
    pascal: Option<&'a [Vec<I>]>,
    _phantom: PhantomData<I>,
}

impl<I, R> BallotCounter<'_, I, R>
where
    I: Integer,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I>,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    /// Processes a ballot, using a recursive method (consistent with Droop.py)
    /// to count ballots that contain candidates ranked equally.
    fn process_ballot_rec(mut self) {
        let voting_power = R::one();
        let ballot_count = I::from_usize(self.ballot.count());
        if !self.ballot.has_tie() {
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

        for ranking in self.ballot.order() {
            // Only one candidate at this level. Easy case.
            let candidate = ranking[0].into();
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
        if i == self.ballot.order_len() {
            return;
        }

        // Fetch the i-th ranking in the ballot, which may contain multiple tied
        // candidates.
        let ranking = self.ballot.order_at(i);

        // TODO: skip defeated candidates according to their status.
        // Number of eligible candidates (with a positive keep factor) in the current
        // ranking.
        let filtered_ranking_len = ranking
            .iter()
            .filter(|&&candidate| !self.keep_factors[candidate.into()].is_zero())
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
        for candidate in ranking.iter().filter_map(|&candidate| {
            let candidate: usize = candidate.into();
            if self.keep_factors[candidate].is_zero() {
                None
            } else {
                Some(candidate)
            }
        }) {
            let (used_power, remaining_power) =
                self.split_power_one_candidate(&voting_power, candidate);
            self.increment_candidate(candidate, used_power, ballot_count);
            if !remaining_power.is_zero() {
                self.process_ballot_rec_impl(remaining_power, ballot_count, i + 1)
            }
        }
    }

    /// Processes a ballot which contains at least a tie, according to the
    /// following "equalized counting" rules.
    ///
    /// In this mode, candidates ranked equally are counted by simulating a
    /// superposition of all possible permutations of these equally-ranked
    /// candidates.
    ///
    /// For example, the ballot `a b=c` becomes a superposition of `a b c` (with
    /// weight 1/2) and `a c b` (with weight 1/2). Likewise, the ballot `a
    /// b=c=d` is counted as a superposition of 6 ballots, each with weight
    /// 1/6: `a b c d`, `a b d c`, `a c b d`, `a c d b`, `a d b c`, `a d c
    /// b`.
    fn process_ballot_equalize(mut self) {
        // Note: we separate the voting power and the number of occurrences of each
        // ballot, and multiply by the ballot count only at the end, so that in
        // order to ensure fairness a repeated ballot is really counted as the sum
        // of `b.count` identical ballots.
        let mut voting_power = R::one();
        let ballot_count = I::from_usize(self.ballot.count());

        let pascal: &[Vec<I>] = self.pascal.unwrap();

        for ranking in self.ballot.order() {
            if ranking.len() == 1 {
                // Only one candidate at this level. Easy case.
                let candidate = ranking[0].into();
                let (used_power, remaining_power) =
                    self.split_power_one_candidate(&voting_power, candidate);
                self.increment_candidate(candidate, used_power, &ballot_count);
                voting_power = remaining_power;
            } else {
                // Multiple candidates ranked equally. Finer computation.
                trace!(
                    "  Ranking = {:?}",
                    ranking.iter().map(|&x| x.into()).collect::<Vec<_>>()
                );
                let ranking_keep_factors: Vec<R> = ranking
                    .iter()
                    .map(|&candidate| self.keep_factors[candidate.into()].clone())
                    .collect();
                // For each candidate, the unused factor is one minus the keep factor (whatever
                // is not kept).
                let ranking_unused_factors: Vec<R> =
                    ranking_keep_factors.iter().map(|k| R::one() - k).collect();
                trace!("    Ranking keep_factors = {ranking_keep_factors:?}");
                for (i, candidate) in ranking.iter().map(|&x| x.into()).enumerate() {
                    let w =
                        Self::polyweights(&ranking_unused_factors, i, &pascal[ranking.len() - 1])
                            * &ranking_keep_factors[i]
                            / I::from_usize(ranking.len());
                    trace!("    polyweight[{i}] = {w} ~ {}", w.to_f64());
                    let used_power = &voting_power * &w;
                    trace!(
                        "  Sum[{candidate}] += {used_power} ~ {}",
                        used_power.to_f64()
                    );
                    self.increment_candidate(candidate, used_power, &ballot_count);
                }
                // Conservatively, the remaining voting power passed to the next ranking in the
                // ballot is the product of the unused factors (rounded down). Like for
                // splitting the voting power for one candidate, we cannot simply let the
                // remaining power being 1 minus the used power, because that would round up the
                // remaining power and potentially giving more votes to candidates further in
                // the ballot.
                // TODO: Multiplication is not associative! Calculate this product in a
                // deterministic and fair way.
                let remaining_power: R = ranking_unused_factors.into_iter().product();
                voting_power *= remaining_power;
            }
            if voting_power.is_zero() {
                break;
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

        if log_enabled!(Trace) {
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
        }

        (used_power, remaining_power)
    }

    /// Helper function to calculate the voting power kept by a candidate in an
    /// equal ranking, according to the "equalized couting" rules.
    ///
    /// Parameters:
    /// - `ranking_unused_factors`: unused factors of the candidates in the
    ///   equal ranking (for each candidate, the unused factor is 1 minus the
    ///   keep factor).
    /// - `skip`: index of the candidate for which the weight is calculated by
    ///   this function.
    /// - `pascal_row`: row of Pascal's triangle with `n` elements, where `n` is
    ///   the length of the `ranking_unused_factors` array.
    #[allow(clippy::needless_range_loop)]
    fn polyweights(ranking_unused_factors: &[R], skip: usize, pascal_row: &[I]) -> R {
        let n = ranking_unused_factors.len() - 1;
        let poly = expand_polynomial(ranking_unused_factors, skip);

        let mut result = R::zero();
        for i in 0..=n {
            result += &poly[i] / &pascal_row[i];
        }
        result
    }
}

/// Computes the coefficients of the polynomial `(X + a)*(X + b)*...*(X +
/// z)`, where `a`, `b`, ..., `z` are the input `coefficients` excluding the
/// one at index `skip`.
///
/// The output poynomial is represented with the most-significant coefficient
/// first, i.e. the entry at index `i` in the output array represents the
/// coefficient of degree `n - i`. In other words, the output array `p`
/// represents the polynomial `p[0]*X^n + p[1]*X^n-1 + ... + p[n-1]*X + p[n]`.
///
/// Warning: even though all the input coefficients should in principle play a
/// symmetric role (except the skipped one), in practice the output will depend
/// on the order of the input coefficients, if multiplication in [`R`] is not
/// associative (e.g. due to rounding). For example, the last output coefficient
/// `p[n]` is computed as `a * b * ... * z` (in this order), so re-ordering `(a,
/// b, ..., z)` can lead to a different result in case of non-associativity.
#[allow(clippy::needless_range_loop)]
fn expand_polynomial<I, R>(coefficients: &[R], skip: usize) -> Vec<R>
where
    I: Integer,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I>,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    // One coefficient is excluded, so the degree `n` of the output polynomial is
    // one less of that.
    let n = coefficients.len() - 1;
    // For a polynomial of degree `n`, there are `n + 1` coefficients.
    let mut poly = vec![R::zero(); n + 1];
    // We start the computation with a polynomial equal to 1.
    poly[0] = R::one();

    // Each iteration of the loop multiplies the current polynomial by (X + a),
    // given the coefficient a.
    for (i, a) in coefficients.iter().enumerate() {
        if i == skip {
            continue;
        }

        // Optimization: do nothing if a coefficient is zero.
        if !a.is_zero() {
            // We start from `P = p[0] X^k + p[1] X^k-1 + ... + p[k]` and want to compute
            // `Q = P * (X + a)`. The resulting coefficients are given by:
            //
            // ```
            //    p[0] X^k+1 +   p[1] X^k + ... +     p[k] X
            // +               a*p[0] X^k + ... + a*p[k-1] X + a*p[k]
            // ------------------------------------------------------
            // =  q[0] X^k+1 +   q[1] X^k + ... +     q[k] X + q[k+1]
            // ```
            //
            // In other words, for each entry q[j], we take p[j], and add to it p[j-1] * a.
            // We do that in one pass with a simple loop.
            let mut prev = poly[0].clone();
            for j in 1..=n {
                let tmp = poly[j].clone();
                poly[j] += prev * a;
                prev = tmp;
            }
        }
    }

    poly
}

/// Computes rows `0..=n` of
/// [Pascal's triangle](https://en.wikipedia.org/wiki/Pascal%27s_triangle),
/// defined by the relation `pascal[n][k] = pascal[n-1][k-1] + pascal[n-1][k]`.
pub fn pascal<I>(n: usize) -> Vec<Vec<I>>
where
    I: Integer,
    for<'a> &'a I: IntegerRef<I>,
{
    let mut result = Vec::new();

    let mut row = vec![I::zero(); n + 1];
    row[0] = I::one();
    result.push(row);

    for i in 1..=n {
        let row = result.last().unwrap();
        let mut newrow = vec![I::zero(); n + 1];
        newrow[0] = I::one();
        for j in 1..=i {
            newrow[j] = &row[j - 1] + &row[j];
        }
        result.push(newrow);
    }
    result
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::arithmetic::{ApproxRational, BigFixedDecimal9, FixedDecimal9, Integer64};
    use crate::types::Candidate;
    use ::test::Bencher;
    use num::rational::Ratio;
    use num::{BigInt, BigRational};
    use std::borrow::Borrow;
    use std::fmt::{Debug, Display};
    use std::hint::black_box;

    macro_rules! numeric_tests {
        ( $typei:ty, $typer:ty, ) => {};
        ( $typei:ty, $typer:ty, $case:ident, $( $others:tt )* ) => {
            #[test]
            fn $case() {
                $crate::vote_count::test::NumericTests::<$typei, $typer>::$case();
            }

            numeric_tests!($typei, $typer, $($others)*);
        };
        ( $typei:ty, $typer:ty, $case:ident => fail($msg:expr), $( $others:tt )* ) => {
            #[test]
            #[should_panic(expected = $msg)]
            fn $case() {
                $crate::vote_count::test::NumericTests::<$typei, $typer>::$case();
            }

            numeric_tests!($typei, $typer, $($others)*);
        };
    }

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

    macro_rules! all_numeric_benches {
        ( $typei:ty, $typer:ty ) => {
            numeric_benches!(
                $typei,
                $typer,
                bench_count_votes_serial_16,
                bench_count_votes_serial_32,
                bench_count_votes_serial_64,
                bench_count_votes_serial_128,
                bench_count_votes_parallel_16,
                bench_count_votes_parallel_32,
                bench_count_votes_parallel_64,
                bench_count_votes_parallel_128,
                bench_process_ballot_rec_chain,
                bench_process_ballot_rec_pairs_01,
                bench_process_ballot_rec_pairs_02,
                bench_process_ballot_rec_pairs_03,
                bench_process_ballot_rec_pairs_04,
                bench_process_ballot_rec_pairs_05,
                bench_process_ballot_rec_pairs_06,
                bench_process_ballot_rec_pairs_07,
                bench_process_ballot_rec_pairs_08,
                bench_process_ballot_rec_pairs_09,
                bench_process_ballot_rec_pairs_10,
                bench_process_ballot_rec_tens_1,
                bench_process_ballot_rec_tens_2,
                bench_process_ballot_rec_tens_3,
                bench_process_ballot_rec_tens_4,
                bench_process_ballot_equalize_chain,
                bench_process_ballot_equalize_pairs_01,
                bench_process_ballot_equalize_pairs_02,
                bench_process_ballot_equalize_pairs_03,
                bench_process_ballot_equalize_pairs_04,
                bench_process_ballot_equalize_pairs_05,
                bench_process_ballot_equalize_pairs_06,
                bench_process_ballot_equalize_pairs_07,
                bench_process_ballot_equalize_pairs_08,
                bench_process_ballot_equalize_pairs_09,
                bench_process_ballot_equalize_pairs_10,
                bench_process_ballot_equalize_pairs_15,
                bench_process_ballot_equalize_tens_1,
                bench_process_ballot_equalize_tens_2,
                bench_process_ballot_equalize_tens_3,
                bench_process_ballot_equalize_tens_4,
            );
        };
    }

    macro_rules! base_numeric_tests {
        ( $typei:ty, $typer:ty ) => {
            numeric_tests!(
                $typei,
                $typer,
                test_write_stats,
                test_threshold,
                test_threshold_exhausted,
                test_surplus_droop,
                test_surplus_positive,
                test_process_ballot_first,
                test_process_ballot_chain,
                test_process_ballot_defeated,
                test_process_ballot_rec_tie_first,
                test_process_ballot_rec_tie_chain,
                test_process_ballot_rec_tie_defeated,
                test_process_ballot_rec_ties,
                test_process_ballot_multiplier,
                test_increment_candidate_ballot_multiplier,
                test_pascal_small,
                test_pascal_50,
                test_expand_polynomial,
                test_polyweights,
            );
        };
    }

    macro_rules! advanced_numeric_tests {
        ( $typei:ty, $typer:ty ) => {
            numeric_tests!(
                $typei,
                $typer,
                test_count_votes_rayon_is_consistent,
                test_count_votes_parallel_is_consistent,
            );
        };
    }

    macro_rules! all_numeric_tests {
        ( $mod:ident, $typei:ty, $typer:ty, $( $other_tests:tt )* ) => {
            mod $mod {
                use super::*;

                base_numeric_tests!($typei, $typer);
                advanced_numeric_tests!($typei, $typer);
                numeric_tests!($typei, $typer, $($other_tests)*);
                all_numeric_benches!($typei, $typer);
            }
        };
    }

    all_numeric_tests!(exact, BigInt, BigRational, test_pascal_100,);
    all_numeric_tests!(approx_rational, BigInt, ApproxRational, test_pascal_100,);
    #[cfg(not(any(feature = "checked_i64", overflow_checks)))]
    all_numeric_tests!(fixed, Integer64, FixedDecimal9, test_pascal_100,);
    #[cfg(feature = "checked_i64")]
    all_numeric_tests!(
        fixed,
        Integer64,
        FixedDecimal9,
        test_pascal_100 => fail(r"called `Option::unwrap()` on a `None` value"),
    );
    #[cfg(all(not(feature = "checked_i64"), overflow_checks))]
    all_numeric_tests!(
        fixed,
        Integer64,
        FixedDecimal9,
        test_pascal_100 => fail(r"attempt to add with overflow"),
    );
    all_numeric_tests!(fixed_big, BigInt, BigFixedDecimal9, test_pascal_100,);

    mod ratio_i64 {
        use super::*;
        base_numeric_tests!(i64, Ratio<i64>);
        #[cfg(not(overflow_checks))]
        numeric_tests!(i64, Ratio<i64>, test_pascal_100,);
        #[cfg(not(overflow_checks))]
        all_numeric_benches!(i64, Ratio<i64>);
    }

    mod float64 {
        all_numeric_benches!(f64, f64);
        numeric_tests!(f64, f64, test_pascal_100,);
    }

    pub struct NumericTests<I, R> {
        _phantomi: PhantomData<I>,
        _phantomr: PhantomData<R>,
    }

    impl<I, R> NumericTests<I, R>
    where
        I: Integer + Send + Sync + Display + Debug + PartialEq,
        for<'a> &'a I: IntegerRef<I>,
        R: Rational<I> + Send + Sync,
        for<'a> &'a R: RationalRef<&'a I, R>,
    {
        fn test_write_stats() {
            let mut buf = Vec::new();

            let vote_count = VoteCount {
                sum: (0..10).map(R::from_usize).collect(),
                exhausted: R::from_usize(42),
                _phantom: PhantomData,
            };
            vote_count
                .write_stats(
                    &mut buf,
                    /* threshold = */ &R::from_usize(123),
                    /* surplus = */ &R::from_usize(456),
                )
                .unwrap();

            let stdout = std::str::from_utf8(&buf).unwrap();
            let expected = format!(
                "\tQuota: {quota}\n\tVotes: {votes}\n\tResidual: {residual}\n\tTotal: {total}\n\tSurplus: {surplus}\n",
                quota = R::from_usize(123),
                votes = R::from_usize(45),
                residual = R::from_usize(42),
                total = R::from_usize(87),
                surplus = R::from_usize(456)
            );
            assert_eq!(stdout, expected);
        }

        fn test_threshold() {
            for num_seats in 0..10 {
                for num_ballots in 0..100 {
                    let election = Election::builder()
                        .title("")
                        .num_seats(num_seats)
                        .num_ballots(num_ballots)
                        .build();

                    let vote_count = VoteCount {
                        sum: Vec::new(),
                        exhausted: R::zero(),
                        _phantom: PhantomData,
                    };
                    assert_eq!(
                        vote_count.threshold(&election),
                        R::ratio(num_ballots, num_seats + 1) + R::epsilon()
                    );

                    let exhausted = std::cmp::min(num_ballots, 42);
                    let vote_count = VoteCount {
                        sum: Vec::new(),
                        exhausted: R::from_usize(exhausted),
                        _phantom: PhantomData,
                    };
                    assert_eq!(
                        vote_count.threshold(&election),
                        R::ratio(num_ballots - exhausted, num_seats + 1) + R::epsilon()
                    );
                }
            }
        }

        fn test_threshold_exhausted() {
            for num_seats in 0..10 {
                for num_ballots in 0..100 {
                    let election = Election::builder()
                        .title("")
                        .num_seats(num_seats)
                        .num_ballots(num_ballots)
                        .build();

                    assert_eq!(
                        VoteCount::threshold_exhausted(
                            &election,
                            /* exhausted = */ &R::zero()
                        ),
                        R::ratio(num_ballots, num_seats + 1) + R::epsilon()
                    );

                    let exhausted = std::cmp::min(num_ballots, 42);
                    assert_eq!(
                        VoteCount::threshold_exhausted(
                            &election,
                            /* exhausted = */ &R::from_usize(exhausted)
                        ),
                        R::ratio(num_ballots - exhausted, num_seats + 1) + R::epsilon()
                    );
                }
            }
        }

        fn test_surplus_droop() {
            let vote_count = VoteCount {
                sum: vec![
                    R::ratio(14, 100),
                    R::ratio(28, 100),
                    R::ratio(57, 100),
                    R::ratio(42, 100),
                    R::ratio(85, 100),
                    R::ratio(71, 100),
                ],
                exhausted: R::zero(),
                _phantom: PhantomData,
            };

            assert_eq!(
                vote_count.surplus_droop(&R::ratio(10, 100), &[0]),
                R::ratio(4, 100)
            );

            assert_eq!(
                vote_count.surplus_droop(&R::ratio(10, 100), &[0, 1, 2, 3, 4, 5]),
                R::ratio(237, 100)
            );
            assert_eq!(
                vote_count.surplus_droop(&R::ratio(40, 100), &[2, 3, 4, 5]),
                R::ratio(95, 100)
            );
            assert_eq!(
                vote_count.surplus_droop(&R::ratio(40, 100), &[0, 1, 2, 3, 4, 5]),
                R::ratio(57, 100)
            );
        }

        fn test_surplus_positive() {
            let vote_count = VoteCount {
                sum: vec![
                    R::ratio(14, 100),
                    R::ratio(28, 100),
                    R::ratio(57, 100),
                    R::ratio(42, 100),
                    R::ratio(85, 100),
                    R::ratio(71, 100),
                ],
                exhausted: R::zero(),
                _phantom: PhantomData,
            };

            assert_eq!(
                vote_count.surplus_positive(&R::ratio(10, 100), &[0]),
                R::ratio(4, 100)
            );

            assert_eq!(
                vote_count.surplus_positive(&R::ratio(10, 100), &[0, 1, 2, 3, 4, 5]),
                R::ratio(237, 100)
            );
            assert_eq!(
                vote_count.surplus_positive(&R::ratio(40, 100), &[2, 3, 4, 5]),
                R::ratio(95, 100)
            );
            assert_eq!(
                vote_count.surplus_positive(&R::ratio(40, 100), &[0, 1, 2, 3, 4, 5]),
                R::ratio(95, 100)
            );
        }

        fn fake_ballots(n: usize) -> Vec<Ballot> {
            let mut ballots = Vec::new();
            for i in 0..n {
                ballots.push(Ballot::new(1, [vec![i]]));
                for j in 0..i {
                    ballots.push(Ballot::new(1, [vec![i], vec![j]]));
                }
            }
            ballots
        }

        fn fake_keep_factors(n: usize) -> Vec<R> {
            (1..=n).map(|i| R::ratio(1, i + 1)).collect()
        }

        fn fake_candidates(n: usize) -> Vec<Candidate> {
            (0..n)
                .map(|i| Candidate::new(format!("candidate{i}"), false))
                .collect()
        }

        fn test_count_votes_rayon_is_consistent() {
            for n in [1, 2, 4, 8, 16, 32, 64, 128] {
                let ballots = Self::fake_ballots(n);
                let keep_factors = Self::fake_keep_factors(n);
                let election = Election::builder()
                    .title("")
                    .candidates(Self::fake_candidates(n))
                    .num_seats(0)
                    .ballots(ballots)
                    .build();

                let vote_count = VoteCount::<I, R>::count_votes(
                    &election,
                    &keep_factors,
                    Parallel::No,
                    None,
                    None,
                );
                let vote_count_parallel = VoteCount::<I, R>::count_votes(
                    &election,
                    &keep_factors,
                    Parallel::Rayon,
                    None,
                    None,
                );
                assert_eq!(
                    vote_count, vote_count_parallel,
                    "Mismatch with {n} candidates"
                );
            }
        }

        fn test_count_votes_parallel_is_consistent() {
            for n in [1, 2, 4, 8, 16, 32, 64, 128] {
                let ballots = Self::fake_ballots(n);
                let keep_factors = Self::fake_keep_factors(n);
                let election = Election::builder()
                    .title("")
                    .candidates(Self::fake_candidates(n))
                    .num_seats(0)
                    .ballots(ballots)
                    .build();

                let vote_count = VoteCount::<I, R>::count_votes(
                    &election,
                    &keep_factors,
                    Parallel::No,
                    None,
                    None,
                );

                for num_threads in 1..=10 {
                    std::thread::scope(|thread_scope| {
                        let thread_pool = VoteCountingThreadPool::new(
                            thread_scope,
                            NonZeroUsize::new(num_threads).unwrap(),
                            RangeStrategy::WorkStealing,
                            &election,
                            None,
                        );
                        let vote_count_parallel = VoteCount::<I, R>::count_votes(
                            &election,
                            &keep_factors,
                            Parallel::Custom,
                            Some(&thread_pool),
                            None,
                        );
                        assert_eq!(
                            vote_count, vote_count_parallel,
                            "Mismatch with {n} candidates"
                        );
                    });
                }
            }
        }

        fn process_ballot_rec(
            ballot: impl Borrow<Ballot>,
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
                pascal: None,
                _phantom: PhantomData,
            };
            counter.process_ballot_rec();

            (sum, unused_power, fn_calls)
        }

        fn process_ballot_equalize(
            ballot: impl Borrow<Ballot>,
            keep_factors: &[R],
            pascal: &[Vec<I>],
        ) -> (Vec<R>, R) {
            let num_candidates = keep_factors.len();
            let mut sum = vec![R::zero(); num_candidates];
            let mut unused_power = R::one();

            let counter = BallotCounter {
                ballot: ballot.borrow(),
                sum: &mut sum,
                unused_power: &mut unused_power,
                keep_factors,
                fn_calls: &mut 0,
                pascal: Some(pascal),
                _phantom: PhantomData,
            };
            counter.process_ballot_equalize();

            (sum, unused_power)
        }

        // First candidate takes it all.
        fn test_process_ballot_first() {
            let ballot = Ballot::new(1, [vec![0], vec![1], vec![2]]);
            let keep_factors = [R::one(), R::one(), R::one()];
            let pascal = pascal::<I>(3);

            // Recursive method.
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(&ballot, &keep_factors);
            assert_eq!(sum, vec![R::one(), R::zero(), R::zero()]);
            assert_eq!(unused_power, R::zero());
            assert_eq!(fn_calls, 1);

            // Equalized counting method.
            let (sum, unused_power) =
                Self::process_ballot_equalize(&ballot, &keep_factors, &pascal);
            assert_eq!(sum, vec![R::one(), R::zero(), R::zero()]);
            assert_eq!(unused_power, R::zero());
        }

        // Chain of keep factors.
        fn test_process_ballot_chain() {
            let ballot = Ballot::new(1, [vec![0], vec![1], vec![2]]);
            let keep_factors = [R::ratio(1, 2), R::ratio(2, 3), R::ratio(3, 4)];
            let pascal = pascal::<I>(3);

            // Recursive method.
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(&ballot, &keep_factors);
            assert_eq!(sum, vec![R::ratio(1, 2), R::ratio(1, 3), R::ratio(1, 8)]);
            assert_eq!(unused_power, R::one() - R::ratio(23, 24));
            assert_eq!(fn_calls, 1);

            // Equalized counting method.
            let (sum, unused_power) =
                Self::process_ballot_equalize(&ballot, &keep_factors, &pascal);
            assert_eq!(sum, vec![R::ratio(1, 2), R::ratio(1, 3), R::ratio(1, 8)]);
            assert_eq!(unused_power, R::one() - R::ratio(23, 24));
        }

        // Defeated candidate (keep factor is zero).
        fn test_process_ballot_defeated() {
            let ballot = Ballot::new(1, [vec![0], vec![1], vec![2]]);
            let keep_factors = [R::zero(), R::ratio(2, 3), R::ratio(3, 4)];
            let pascal = pascal::<I>(3);

            // Recursive method.
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(&ballot, &keep_factors);
            assert_eq!(sum, vec![R::zero(), R::ratio(2, 3), R::ratio(1, 4)]);
            assert_eq!(unused_power, R::one() - R::ratio(11, 12));
            assert_eq!(fn_calls, 1);

            // Equalized counting method.
            let (sum, unused_power) =
                Self::process_ballot_equalize(&ballot, &keep_factors, &pascal);
            assert_eq!(sum, vec![R::zero(), R::ratio(2, 3), R::ratio(1, 4)]);
            assert_eq!(unused_power, R::one() - R::ratio(11, 12));
        }

        // Tie to start the ballot.
        fn test_process_ballot_rec_tie_first() {
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(
                Ballot::new(1, [vec![0, 1], vec![2]]),
                /* keep_factors = */ &[R::one(), R::one(), R::one()],
            );

            assert_eq!(sum, vec![R::ratio(1, 2), R::ratio(1, 2), R::zero()]);
            assert_eq!(unused_power, R::zero());
            assert_eq!(fn_calls, 1);
        }

        // Tie with chaining.
        fn test_process_ballot_rec_tie_chain() {
            let (sum, unused_power, fn_calls) = Self::process_ballot_rec(
                Ballot::new(1, [vec![0, 1], vec![2]]),
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
                Ballot::new(1, [vec![0], vec![1, 2]]),
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
                Ballot::new(1, [vec![0, 1], vec![2, 3]]),
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

        fn test_process_ballot_multiplier() {
            let ballots = [
                vec![vec![0], vec![1], vec![2], vec![3], vec![4]],
                vec![vec![4], vec![3], vec![2], vec![1], vec![0]],
                vec![vec![0, 2], vec![1, 3, 4]],
                vec![vec![0, 1, 2, 3, 4]],
            ];
            let keep_factors_list = [
                [R::zero(), R::zero(), R::zero(), R::zero(), R::zero()],
                [R::one(), R::one(), R::one(), R::one(), R::one()],
                [
                    R::ratio(1, 2),
                    R::ratio(2, 3),
                    R::ratio(4, 5),
                    R::ratio(6, 7),
                    R::ratio(10, 11),
                ],
            ];
            for keep_factors in &keep_factors_list {
                for order in &ballots {
                    for ballot_count in 1..=30 {
                        // Count the ballot once with a multiplier of ballot_count.
                        let left_vote_count = {
                            let mut vote_accumulator = VoteAccumulator::new(5);
                            let ballot = Ballot::new(ballot_count, order.clone());
                            VoteCount::process_ballot(
                                &mut vote_accumulator,
                                keep_factors,
                                None,
                                0,
                                &ballot,
                            );
                            vote_accumulator.into_vote_count()
                        };

                        // Count the ballot ballot_count times with a multiplier of one each time.
                        let right_vote_count = {
                            let mut vote_accumulator = VoteAccumulator::new(5);
                            let ballot = Ballot::new(1, order.clone());
                            for _ in 0..ballot_count {
                                VoteCount::process_ballot(
                                    &mut vote_accumulator,
                                    keep_factors,
                                    None,
                                    0,
                                    &ballot,
                                );
                            }
                            vote_accumulator.into_vote_count()
                        };

                        // Check that both match.
                        assert_eq!(left_vote_count.sum, right_vote_count.sum);
                        assert_eq!(left_vote_count.exhausted, right_vote_count.exhausted);
                    }
                }
            }
        }

        fn test_increment_candidate_ballot_multiplier() {
            let empty_ballot = Ballot::empty();
            let empty_keep_factors = vec![];

            for used_power in R::get_positive_test_values() {
                for ballot_count in 1..=30 {
                    // Increment the candidate once with a multiplier of ballot_count.
                    let (left_sum, left_unused_power) = {
                        let mut sum = vec![R::zero()];
                        let mut unused_power = R::zero();
                        let mut ballot_counter = BallotCounter {
                            ballot: &empty_ballot,
                            sum: &mut sum,
                            unused_power: &mut unused_power,
                            keep_factors: &empty_keep_factors,
                            fn_calls: &mut 0,
                            pascal: None,
                            _phantom: PhantomData,
                        };
                        ballot_counter.increment_candidate(
                            0,
                            used_power.clone(),
                            &I::from_usize(ballot_count),
                        );
                        (sum, unused_power)
                    };

                    // Increment the candidate ballot_count times, with a multiplier of one each
                    // time.
                    let (right_sum, right_unused_power) = {
                        let mut sum = vec![R::zero()];
                        let mut unused_power = R::zero();
                        let mut ballot_counter = BallotCounter {
                            ballot: &empty_ballot,
                            sum: &mut sum,
                            unused_power: &mut unused_power,
                            keep_factors: &empty_keep_factors,
                            fn_calls: &mut 0,
                            pascal: None,
                            _phantom: PhantomData,
                        };
                        for _ in 0..ballot_count {
                            ballot_counter.increment_candidate(
                                0,
                                used_power.clone(),
                                &I::from_usize(1),
                            );
                        }
                        (sum, unused_power)
                    };

                    // Check that something was incremented.
                    assert_ne!(left_sum[0], R::zero());
                    assert_ne!(left_unused_power, R::zero());

                    // Check that both match.
                    assert_eq!(left_sum, right_sum);
                    assert_eq!(left_unused_power, right_unused_power);
                }
            }
        }

        fn test_pascal_small() {
            assert_eq!(
                pascal::<I>(0),
                [[1]].map(|row| row.map(I::from_usize).to_vec()).to_vec()
            );
            assert_eq!(
                pascal::<I>(1),
                [[1, 0], [1, 1]]
                    .map(|row| row.map(I::from_usize).to_vec())
                    .to_vec()
            );
            assert_eq!(
                pascal::<I>(5),
                [
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 2, 1, 0, 0, 0],
                    [1, 3, 3, 1, 0, 0],
                    [1, 4, 6, 4, 1, 0],
                    [1, 5, 10, 10, 5, 1],
                ]
                .map(|row| row.map(I::from_usize).to_vec())
                .to_vec()
            );
            assert_eq!(
                pascal::<I>(10),
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0],
                    [1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0],
                    [1, 6, 15, 20, 15, 6, 1, 0, 0, 0, 0],
                    [1, 7, 21, 35, 35, 21, 7, 1, 0, 0, 0],
                    [1, 8, 28, 56, 70, 56, 28, 8, 1, 0, 0],
                    [1, 9, 36, 84, 126, 126, 84, 36, 9, 1, 0],
                    [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1],
                ]
                .map(|row| row.map(I::from_usize).to_vec())
                .to_vec()
            );
        }

        fn test_pascal_50() {
            Self::test_pascal_properties(50);
        }

        fn test_pascal_100() {
            Self::test_pascal_properties(100);
        }

        fn test_pascal_properties(count: usize) {
            let pascal = pascal::<I>(count);
            for i in 0..=count {
                assert_eq!(pascal[i][0], I::from_usize(1));
                assert_eq!(pascal[i][i], I::from_usize(1));
                assert_eq!(pascal[i][1], I::from_usize(i));
                if i < count {
                    assert_eq!(pascal[i + 1][i], I::from_usize(i + 1));
                }
                for j in 1..=i {
                    assert_eq!(pascal[i][j], &pascal[i - 1][j - 1] + &pascal[i - 1][j]);
                }
            }
        }

        fn test_expand_polynomial() {
            assert_eq!(
                expand_polynomial::<I, R>(&[1].map(R::from_usize), 0),
                [1].map(R::from_usize)
            );
            assert_eq!(
                expand_polynomial::<I, R>(&[1, 1].map(R::from_usize), 0),
                [1, 1].map(R::from_usize)
            );
            assert_eq!(
                expand_polynomial::<I, R>(&[1, 1, 1].map(R::from_usize), 0),
                [1, 2, 1].map(R::from_usize)
            );
            assert_eq!(
                expand_polynomial::<I, R>(&[1, 1, 1, 1].map(R::from_usize), 0),
                [1, 3, 3, 1].map(R::from_usize)
            );
            assert_eq!(
                expand_polynomial::<I, R>(&[1, 1, 1, 1, 1].map(R::from_usize), 0),
                [1, 4, 6, 4, 1].map(R::from_usize)
            );
            // (x + 1)(x + 2)(x + 3)
            // = (x^2 + 3x + 2)(x + 3)
            // = x^3 + 3x^2 + 2x + 3x^2 + 9x + 6
            // = x^3 + 6x^2 + 11x + 6
            assert_eq!(
                expand_polynomial::<I, R>(&[1, 2, 3, 0].map(R::from_usize), 3),
                [1, 6, 11, 6].map(R::from_usize)
            );
            // (x + 2)(x + 3)(x + 4)(x + 5)
            assert_eq!(
                expand_polynomial::<I, R>(&[1, 2, 3, 4, 5].map(R::from_usize), 0),
                [1, 14, 71, 154, 120].map(R::from_usize)
            );
        }

        fn polyweight_array(keep_factors: &[R], pascal: &[Vec<I>]) -> Vec<R> {
            let unused_factors: Vec<_> = keep_factors.iter().map(|k| R::one() - k).collect();
            let count = keep_factors.len();
            (0..count)
                .map(|i| BallotCounter::<I, R>::polyweights(&unused_factors, i, &pascal[count - 1]))
                .collect::<Vec<_>>()
        }

        fn weights(keep_factors: &[R], pascal: &[Vec<I>]) -> Vec<R> {
            let unused_factors: Vec<_> = keep_factors.iter().map(|k| R::one() - k).collect();
            let count = keep_factors.len();
            (0..count)
                .map(|i| {
                    BallotCounter::<I, R>::polyweights(&unused_factors, i, &pascal[count - 1])
                        * &keep_factors[i]
                        / I::from_usize(count)
                })
                .collect::<Vec<_>>()
        }

        fn test_polyweights() {
            let count = 3;
            let pascal = pascal::<I>(count);

            let keep_factors = [1, 1, 1].map(R::from_usize);
            assert_eq!(
                Self::polyweight_array(&keep_factors, &pascal),
                [1, 1, 1].map(R::from_usize)
            );
            assert_eq!(
                Self::weights(&keep_factors, &pascal),
                [R::ratio(1, 3), R::ratio(1, 3), R::ratio(1, 3)]
            );

            let keep_factors = [1, 0, 1].map(R::from_usize);
            assert_eq!(
                Self::polyweight_array(&keep_factors, &pascal),
                [R::ratio(3, 2), R::from_usize(1), R::ratio(3, 2)]
            );
            assert_eq!(
                Self::weights(&keep_factors, &pascal),
                [R::ratio(1, 2), R::from_usize(0), R::ratio(1, 2)]
            );

            let keep_factors = [1, 0, 0].map(R::from_usize);
            assert_eq!(
                Self::polyweight_array(&keep_factors, &pascal),
                [R::from_usize(3), R::ratio(3, 2), R::ratio(3, 2)]
            );
            assert_eq!(
                Self::weights(&keep_factors, &pascal),
                [R::from_usize(1), R::from_usize(0), R::from_usize(0)]
            );

            // ABC + ACB => [1/3, 0, 0]
            // BAC => [1/12, 1/12, 0]
            // CAB => [1/12, 0, 1/12]
            // BCA => [1/24, 1/12, 1/24]
            // CBA => [1/24, 1/24, 1/12]
            // =====
            // Total = [7/12, 5/24, 5/24]
            let keep_factors = [R::from_usize(1), R::ratio(1, 2), R::ratio(1, 2)];
            assert_eq!(
                Self::polyweight_array(&keep_factors, &pascal),
                [R::ratio(7, 4), R::ratio(5, 4), R::ratio(5, 4)]
            );
            assert_eq!(
                Self::weights(&keep_factors, &pascal),
                [R::ratio(7, 12), R::ratio(5, 24), R::ratio(5, 24)]
            );

            // 1/8th of each vote is forwarded to the next rank.
            // The kept 7/8th are split in 3.
            let keep_factors = [R::ratio(1, 2), R::ratio(1, 2), R::ratio(1, 2)];
            assert_eq!(
                Self::polyweight_array(&keep_factors, &pascal),
                [R::ratio(7, 4), R::ratio(7, 4), R::ratio(7, 4)]
            );
            assert_eq!(
                Self::weights(&keep_factors, &pascal),
                [R::ratio(7, 24), R::ratio(7, 24), R::ratio(7, 24)]
            );
        }

        fn bench_count_votes_serial_16(bencher: &mut Bencher) {
            Self::bench_count_votes(bencher, 16, Parallel::No);
        }

        fn bench_count_votes_serial_32(bencher: &mut Bencher) {
            Self::bench_count_votes(bencher, 32, Parallel::No);
        }

        fn bench_count_votes_serial_64(bencher: &mut Bencher) {
            Self::bench_count_votes(bencher, 64, Parallel::No);
        }

        fn bench_count_votes_serial_128(bencher: &mut Bencher) {
            Self::bench_count_votes(bencher, 128, Parallel::No);
        }

        fn bench_count_votes_parallel_16(bencher: &mut Bencher) {
            Self::bench_count_votes(bencher, 16, Parallel::Rayon);
        }

        fn bench_count_votes_parallel_32(bencher: &mut Bencher) {
            Self::bench_count_votes(bencher, 32, Parallel::Rayon);
        }

        fn bench_count_votes_parallel_64(bencher: &mut Bencher) {
            Self::bench_count_votes(bencher, 64, Parallel::Rayon);
        }

        fn bench_count_votes_parallel_128(bencher: &mut Bencher) {
            Self::bench_count_votes(bencher, 128, Parallel::Rayon);
        }

        fn bench_count_votes(bencher: &mut Bencher, n: usize, parallel: Parallel) {
            let ballots = Self::fake_ballots(n);
            let keep_factors = Self::fake_keep_factors(n);
            let election = Election::builder()
                .title("")
                .candidates(Self::fake_candidates(n))
                .num_seats(0)
                .ballots(ballots)
                .build();

            bencher.iter(|| {
                VoteCount::<I, R>::count_votes(
                    black_box(&election),
                    black_box(&keep_factors),
                    parallel,
                    None,
                    None,
                )
            });
        }

        fn bench_process_ballot_rec_chain(bencher: &mut Bencher) {
            let ballot = Ballot::new(1, (0..10).map(|i| vec![i]));
            let keep_factors = Self::fake_keep_factors(10);
            bencher.iter(|| Self::process_ballot_rec(black_box(&ballot), black_box(&keep_factors)))
        }

        fn bench_process_ballot_rec_pairs_01(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_pairs(bencher, 1);
        }

        fn bench_process_ballot_rec_pairs_02(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_pairs(bencher, 2);
        }

        fn bench_process_ballot_rec_pairs_03(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_pairs(bencher, 3);
        }

        fn bench_process_ballot_rec_pairs_04(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_pairs(bencher, 4);
        }

        fn bench_process_ballot_rec_pairs_05(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_pairs(bencher, 5);
        }

        fn bench_process_ballot_rec_pairs_06(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_pairs(bencher, 6);
        }

        fn bench_process_ballot_rec_pairs_07(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_pairs(bencher, 7);
        }

        fn bench_process_ballot_rec_pairs_08(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_pairs(bencher, 8);
        }

        fn bench_process_ballot_rec_pairs_09(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_pairs(bencher, 9);
        }

        fn bench_process_ballot_rec_pairs_10(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_pairs(bencher, 10);
        }

        fn bench_process_ballot_rec_pairs(bencher: &mut Bencher, layers: usize) {
            let n = layers * 2;
            let ballot = Ballot::new(1, (0..n).array_chunks::<2>());
            let keep_factors = Self::fake_keep_factors(n);
            bencher.iter(|| Self::process_ballot_rec(black_box(&ballot), black_box(&keep_factors)))
        }

        fn bench_process_ballot_rec_tens_1(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_tens(bencher, 1);
        }

        fn bench_process_ballot_rec_tens_2(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_tens(bencher, 2);
        }

        fn bench_process_ballot_rec_tens_3(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_tens(bencher, 3);
        }

        fn bench_process_ballot_rec_tens_4(bencher: &mut Bencher) {
            Self::bench_process_ballot_rec_tens(bencher, 4);
        }

        fn bench_process_ballot_rec_tens(bencher: &mut Bencher, layers: usize) {
            let n = layers * 10;
            let ballot = Ballot::new(1, (0..n).array_chunks::<10>());
            let keep_factors = Self::fake_keep_factors(n);
            bencher.iter(|| Self::process_ballot_rec(black_box(&ballot), black_box(&keep_factors)))
        }

        fn bench_process_ballot_equalize_chain(bencher: &mut Bencher) {
            let pascal = pascal::<I>(10);
            let ballot = Ballot::new(1, (0..10).map(|i| vec![i]));
            let keep_factors = Self::fake_keep_factors(10);
            bencher.iter(|| {
                Self::process_ballot_equalize(black_box(&ballot), black_box(&keep_factors), &pascal)
            })
        }

        fn bench_process_ballot_equalize_pairs_01(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 1);
        }

        fn bench_process_ballot_equalize_pairs_02(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 2);
        }

        fn bench_process_ballot_equalize_pairs_03(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 3);
        }

        fn bench_process_ballot_equalize_pairs_04(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 4);
        }

        fn bench_process_ballot_equalize_pairs_05(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 5);
        }

        fn bench_process_ballot_equalize_pairs_06(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 6);
        }

        fn bench_process_ballot_equalize_pairs_07(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 7);
        }

        fn bench_process_ballot_equalize_pairs_08(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 8);
        }

        fn bench_process_ballot_equalize_pairs_09(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 9);
        }

        fn bench_process_ballot_equalize_pairs_10(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 10);
        }

        fn bench_process_ballot_equalize_pairs_15(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_pairs(bencher, 15);
        }

        fn bench_process_ballot_equalize_pairs(bencher: &mut Bencher, layers: usize) {
            let n = layers * 2;
            let pascal = pascal::<I>(n);
            let ballot = Ballot::new(1, (0..n).array_chunks::<2>());
            let keep_factors = Self::fake_keep_factors(n);
            bencher.iter(|| {
                Self::process_ballot_equalize(black_box(&ballot), black_box(&keep_factors), &pascal)
            })
        }

        fn bench_process_ballot_equalize_tens_1(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_tens(bencher, 1);
        }

        fn bench_process_ballot_equalize_tens_2(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_tens(bencher, 2);
        }

        fn bench_process_ballot_equalize_tens_3(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_tens(bencher, 3);
        }

        fn bench_process_ballot_equalize_tens_4(bencher: &mut Bencher) {
            Self::bench_process_ballot_equalize_tens(bencher, 4);
        }

        fn bench_process_ballot_equalize_tens(bencher: &mut Bencher, layers: usize) {
            let n = layers * 10;
            let pascal = pascal::<I>(n);
            let ballot = Ballot::new(1, (0..n).array_chunks::<10>());
            let keep_factors = Self::fake_keep_factors(n);
            bencher.iter(|| {
                Self::process_ballot_equalize(black_box(&ballot), black_box(&keep_factors), &pascal)
            })
        }
    }
}
