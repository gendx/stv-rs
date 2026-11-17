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

//! Single Transferable Vote implementation of Meek's algorithm. This aims to
//! give consistent results with Droop.py.

use crate::arithmetic::{Integer, IntegerRef, Rational, RationalRef};
use crate::cli::Parallel;
use crate::parallelism::RangeStrategy;
use crate::types::{ElectedReason, Election, ElectionResult, NotElectedReason, TranscriptEvent};
use crate::vote_count::{self, VoteCount, VoteCountingThreadPool};
use log::{Level, debug, info, log_enabled, warn};
use std::fmt::{self, Debug, Display};
use std::io;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::time::Instant;

/// Runs an election according to Meek's rules. This aims to produce
/// reproducible results w.r.t. Droop.py.
#[expect(clippy::too_many_arguments)]
pub fn stv_droop<I, R>(
    stdout: &mut impl io::Write,
    election: &Election,
    package_name: &str,
    omega_exponent: usize,
    parallel: Parallel,
    num_threads: Option<NonZeroUsize>,
    disable_work_stealing: bool,
    force_positive_surplus: bool,
    equalize: bool,
) -> io::Result<ElectionResult<R>>
where
    I: Integer + Send + Sync,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I> + Send + Sync,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    info!(
        "Parallel vote counting is {}",
        match parallel {
            Parallel::No => "disabled",
            Parallel::Rayon => "enabled with rayon",
            Parallel::Custom => "enabled using custom threads",
        }
    );
    info!(
        "Equalized vote counting is {}",
        if equalize { "enabled" } else { "disabled" }
    );

    let pascal = if equalize {
        Some(vote_count::pascal::<I>(election.num_candidates))
    } else {
        None
    };
    let pascal = pascal.as_deref();

    match parallel {
        Parallel::No => {
            let state = State::<I, R>::new(
                election,
                omega_exponent,
                parallel,
                force_positive_surplus,
                pascal,
                None,
            );
            state.run(stdout, package_name, omega_exponent)
        }
        Parallel::Rayon => {
            if let Some(num_threads) = num_threads {
                info!("Spawning {num_threads} rayon threads");
                rayon::ThreadPoolBuilder::new()
                    .num_threads(num_threads.into())
                    .build_global()
                    .unwrap();
            }

            let state = State::<I, R>::new(
                election,
                omega_exponent,
                parallel,
                force_positive_surplus,
                pascal,
                None,
            );
            state.run(stdout, package_name, omega_exponent)
        }
        Parallel::Custom => std::thread::scope(|thread_scope| -> io::Result<ElectionResult<R>> {
            let num_threads: NonZeroUsize = num_threads.unwrap_or_else(|| {
                let count = match std::thread::available_parallelism() {
                    Ok(num_threads) => num_threads.get(),
                    Err(e) => {
                        warn!("Failed to query available parallelism: {e:?}");
                        1
                    }
                };
                NonZeroUsize::new(count)
                    .expect("A positive number of threads must be spawned, but the available parallelism is zero threads")
            });
            info!("Spawning {num_threads} threads");
            let thread_pool = VoteCountingThreadPool::new(
                thread_scope,
                num_threads,
                if disable_work_stealing {
                    RangeStrategy::Fixed
                } else {
                    RangeStrategy::WorkStealing
                },
                election,
                pascal,
            );

            let state = State::<I, R>::new(
                election,
                omega_exponent,
                parallel,
                force_positive_surplus,
                pascal,
                Some(thread_pool),
            );
            state.run(stdout, package_name, omega_exponent)
        }),
    }
}

/// Status of a candidate during the vote-counting process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Candidate that can still be either elected or defeated.
    Candidate,
    /// Candidate that has withdrawn from the election, and will not be elected.
    Withdrawn,
    /// Candidate that was elected.
    Elected,
    /// Candidate that was not elected.
    NotElected,
}

/// Result of a Droop iteration.
#[derive(Debug, PartialEq, Eq)]
enum DroopIteration {
    /// A candidate was elected in this iteration.
    Elected,
    /// The iteration reached a stable state of counted votes and keep factors.
    Stable,
    /// The surplus decreased below the omega threshold.
    Omega,
}

impl Display for DroopIteration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DroopIteration::Elected => f.write_str("elected"),
            DroopIteration::Stable => f.write_str("stable"),
            DroopIteration::Omega => f.write_str("omega"),
        }
    }
}

/// Running state while computing the election results.
pub struct State<'scope, 'e, I, R> {
    /// Election input.
    election: &'e Election,
    /// Transcript of the election counting.
    transcript: Vec<TranscriptEvent<R>>,
    /// Status of each candidate.
    statuses: Vec<Status>,
    /// List of elected candidates.
    elected: Vec<usize>,
    /// List of non-elected candidates.
    not_elected: Vec<usize>,
    /// Number of candidates remaining to elect.
    to_elect: usize,
    /// Keep factor of each candidate.
    keep_factors: Vec<R>,
    /// Current threshold for a candidate to be elected.
    threshold: R,
    /// Surplus of voting power distributed to elected candidates, beyond the
    /// required threshold.
    surplus: R,
    /// When the surplus becomes smaller than this paramter, an election
    /// iteration is considered stabilized.
    omega: R,
    /// Whether parallel ballot counting (based on the rayon crate) is enabled.
    parallel: Parallel,
    /// Whether to use a positive surplus calculation (which avoids potential
    /// crashes if the surplus otherwise becomes negative.
    force_positive_surplus: bool,
    /// Pre-computed Pascal triangle. Set only if the "equalized counting" is
    /// enabled.
    pascal: Option<&'e [Vec<I>]>,
    /// Thread pool where to schedule vote counting. Set only if the parallel
    /// mode is set to "custom".
    thread_pool: Option<VoteCountingThreadPool<'scope, I, R>>,
    _phantom: PhantomData<I>,
}

#[cfg(test)]
impl<'e, I, R> State<'_, 'e, I, R> {
    fn builder() -> test::StateBuilder<'e, I, R> {
        test::StateBuilder::default()
    }
}

impl<'scope, 'e, I, R> State<'scope, 'e, I, R>
where
    I: Integer + Send + Sync + 'e,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I> + Send + Sync + 'scope,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    fn new<'a>(
        election: &'a Election,
        omega_exponent: usize,
        parallel: Parallel,
        force_positive_surplus: bool,
        pascal: Option<&'a [Vec<I>]>,
        thread_pool: Option<VoteCountingThreadPool<'scope, I, R>>,
    ) -> State<'scope, 'a, I, R> {
        let mut transcript = Vec::new();
        let statuses: Vec<_> = election
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if c.is_withdrawn {
                    transcript.push(TranscriptEvent::Withdrawn(i));
                    Status::Withdrawn
                } else {
                    Status::Candidate
                }
            })
            .collect();
        let keep_factors = vec![R::one(); election.candidates.len()];
        transcript.push(TranscriptEvent::KeepFactors(keep_factors.clone()));
        State {
            election,
            transcript,
            statuses,
            elected: Vec::new(),
            not_elected: Vec::new(),
            to_elect: election.num_seats,
            keep_factors,
            threshold: VoteCount::<I, R>::threshold_exhausted(election, &R::zero()),
            surplus: R::zero(),
            omega: R::one() / (0..omega_exponent).map(|_| I::from_usize(10)).product(),
            parallel,
            force_positive_surplus,
            pascal,
            thread_pool,
            _phantom: PhantomData,
        }
    }

    /// Runs the main loop of Meek's rules.
    fn run(
        mut self,
        stdout: &mut impl io::Write,
        package_name: &str,
        omega_exponent: usize,
    ) -> io::Result<ElectionResult<R>> {
        let beginning = Instant::now();
        let mut timestamp = beginning;

        let mut count = self.start_election(stdout, package_name, omega_exponent)?;
        self.transcript.push(TranscriptEvent::Count {
            tallies: count.sum.clone(),
            exhausted: count.exhausted.clone(),
        });

        for round in 1.. {
            if self.election_completed(stdout, round)? {
                break;
            }

            self.transcript.push(TranscriptEvent::StartRound(round));
            self.election_round(stdout, &mut count)?;

            if log_enabled!(Level::Debug) {
                let now = Instant::now();
                debug!(
                    "Elapsed time to compute this round: {:?} / total: {:?}",
                    now.duration_since(timestamp),
                    now.duration_since(beginning)
                );
                timestamp = now;
            }
        }

        self.handle_remaining_candidates(stdout, &mut count)?;
        self.transcript.push(TranscriptEvent::Count {
            tallies: count.sum.clone(),
            exhausted: count.exhausted.clone(),
        });

        self.write_action(stdout, "Count Complete", &count, true)?;

        if log_enabled!(Level::Info) {
            let now = Instant::now();
            info!("Total elapsed time: {:?}", now.duration_since(beginning));
        }

        let result = ElectionResult {
            elected: self.elected,
            not_elected: self.not_elected,
            transcript: self.transcript,
        };

        writeln!(stdout)?;
        Ok(result)
    }

    /// Elects a candidate, and updates the state accordingly.
    fn elect_candidate(&mut self, i: usize, reason: ElectedReason) {
        info!(
            "Electing candidate #{i} ({})",
            self.election.candidates[i].nickname
        );
        assert!(self.to_elect != 0);
        self.elected.push(i);
        self.statuses[i] = Status::Elected;
        self.to_elect -= 1;
        self.transcript.push(TranscriptEvent::Elected(i, reason));
    }

    /// Defeats a candidate, and updates the state accordingly.
    fn defeat_candidate(&mut self, i: usize, reason: NotElectedReason) {
        info!(
            "Defeating candidate #{i} ({})",
            self.election.candidates[i].nickname
        );
        self.not_elected.push(i);
        self.statuses[i] = Status::NotElected;
        self.transcript.push(TranscriptEvent::NotElected(i, reason));
    }

    /// Returns the number of candidates.
    fn num_candidates(&self) -> usize {
        self.election.num_candidates
    }

    /// Counts the ballots based on the current keep factors.
    fn count_votes(&self) -> VoteCount<I, R> {
        VoteCount::<I, R>::count_votes(
            self.election,
            &self.keep_factors,
            self.parallel,
            self.thread_pool.as_ref(),
            self.pascal,
        )
    }

    /// Performs the initial counting to start the election.
    fn start_election(
        &self,
        stdout: &mut impl io::Write,
        package_name: &str,
        omega_exponent: usize,
    ) -> io::Result<VoteCount<I, R>> {
        writeln!(
            stdout,
            r"
Election: {}

	{package_name}
	Rule: Meek Parametric (omega = 1/10^{omega_exponent})
	Arithmetic: {}
	Seats: {}
	Ballots: {}
	Quota: {}
	Omega: {}
",
            self.election.title,
            R::description(),
            self.election.num_seats,
            self.election.num_ballots,
            self.threshold,
            self.omega,
        )?;

        for candidate in &self.election.candidates {
            writeln!(
                stdout,
                "\tAdd {}: {}",
                if candidate.is_withdrawn {
                    "withdrawn"
                } else {
                    "eligible"
                },
                candidate.name
            )?;
        }

        let mut count = self.count_votes();
        // Droop somehow doesn't report exhausted count on the initial count.
        count.exhausted = R::zero();

        self.write_action(stdout, "Begin Count", &count, true)?;

        Ok(count)
    }

    /// Returns true if the election is complete.
    fn election_completed(&self, stdout: &mut impl io::Write, round: usize) -> io::Result<bool> {
        let candidate_count = self
            .statuses
            .iter()
            .filter(|&&status| status == Status::Candidate)
            .count();
        debug!(
            "Checking if count is complete: candidates={candidate_count}, to_elect={}",
            self.to_elect
        );
        assert!(candidate_count >= self.to_elect);
        if self.to_elect == 0 || candidate_count == self.to_elect {
            info!("Count is now complete!");
            return Ok(true);
        }
        writeln!(stdout, "Round {round}:")?;

        if log_enabled!(Level::Debug) {
            debug!("Weights:");
            for (i, candidate) in self.election.candidates.iter().enumerate() {
                debug!(
                    "    [{i}] {} ({:?}) ~ {}",
                    candidate.nickname,
                    self.statuses[i],
                    self.keep_factors[i].to_f64()
                );
            }
        }

        if self
            .statuses
            .iter()
            .all(|&status| status != Status::Candidate)
        {
            debug!("Election done!");
            return Ok(true);
        }

        Ok(false)
    }

    /// Runs one election round, which either elects or defeats exactly one
    /// candidate.
    fn election_round(
        &mut self,
        stdout: &mut impl io::Write,
        count: &mut VoteCount<I, R>,
    ) -> io::Result<()> {
        let iteration = self.iterate_droop(stdout, count)?;
        self.write_action(stdout, &format!("Iterate ({iteration})"), count, false)?;

        let reason = match iteration {
            DroopIteration::Elected => {
                debug!("Iteration returned Elected, continuing the loop");
                None
            }
            DroopIteration::Stable => Some(NotElectedReason::StableSurplus),
            DroopIteration::Omega => Some(NotElectedReason::LowSurplus),
        };

        if let Some(reason) = reason {
            let not_elected = self.next_defeated_candidate(stdout, count)?;

            self.defeat_candidate(not_elected, reason);
            let reason = match reason {
                NotElectedReason::StableSurplus => format!("stable surplus {}", self.surplus),
                NotElectedReason::LowSurplus => format!("surplus {} < omega", self.surplus),
                NotElectedReason::Remaining | NotElectedReason::Other => unreachable!(),
            };
            let message = format!(
                "Defeat ({reason}): {}",
                self.election.candidates[not_elected].name
            );
            self.write_action(stdout, &message, count, true)?;

            self.keep_factors[not_elected] = R::zero();
            self.transcript
                .push(TranscriptEvent::KeepFactors(self.keep_factors.clone()));

            *count = self.count_votes();
            self.transcript.push(TranscriptEvent::Count {
                tallies: count.sum.clone(),
                exhausted: count.exhausted.clone(),
            });
        }

        self.debug_count(count);

        Ok(())
    }

    /// Iterates the vote counting process until either a candidate is elected,
    /// the surplus becomes lower than omega, or the count has stabilized.
    fn iterate_droop(
        &mut self,
        stdout: &mut impl io::Write,
        count: &mut VoteCount<I, R>,
    ) -> io::Result<DroopIteration> {
        let mut last_surplus = R::from_usize(self.election.num_ballots);

        loop {
            *count = self.count_votes();
            self.transcript.push(TranscriptEvent::Count {
                tallies: count.sum.clone(),
                exhausted: count.exhausted.clone(),
            });

            self.threshold = count.threshold(self.election);
            self.transcript
                .push(TranscriptEvent::Quota(self.threshold.clone()));

            let has_elected = self.elect_candidates(stdout, count)?;

            self.surplus = if self.force_positive_surplus {
                count.surplus_positive(&self.threshold, &self.elected)
            } else {
                count.surplus_droop(&self.threshold, &self.elected)
            };
            self.transcript
                .push(TranscriptEvent::Surplus(self.surplus.clone()));

            if has_elected {
                debug!("Returning from iterate_droop (Elected)");
                return Ok(DroopIteration::Elected);
            }

            if self.surplus <= self.omega {
                debug!("Returning from iterate_droop (Omega)");
                return Ok(DroopIteration::Omega);
            }

            if self.surplus >= last_surplus {
                writeln!(stdout, "\tStable state detected ({})", self.surplus)?;
                debug!("Returning from iterate_droop (Stable)");
                return Ok(DroopIteration::Stable);
            }

            last_surplus = self.surplus.clone();

            // Update keep factors of elected candidates.
            debug!("Updating keep factors");
            for i in self.statuses.iter().enumerate().filter_map(|(i, &status)| {
                if status == Status::Elected {
                    Some(i)
                } else {
                    None
                }
            }) {
                let new_factor = self.keep_factors[i]
                    .mul_up(&self.threshold)
                    .div_up_as_keep_factor(&count.sum[i]);
                debug!(
                    "\t{}: {} ~ {} -> {new_factor} ~ {}",
                    self.election.candidates[i].nickname,
                    self.keep_factors[i],
                    self.keep_factors[i].to_f64(),
                    new_factor.to_f64()
                );
                self.keep_factors[i] = new_factor;
            }
            self.transcript
                .push(TranscriptEvent::KeepFactors(self.keep_factors.clone()));
        }
    }

    /// Elect candidates that exceed the threshold, returning true if any
    /// candidate was elected.
    fn elect_candidates(
        &mut self,
        stdout: &mut impl io::Write,
        count: &VoteCount<I, R>,
    ) -> io::Result<bool> {
        let mut has_elected = false;
        for (i, candidate) in self.election.candidates.iter().enumerate() {
            match self.statuses[i] {
                // Elected candidates must always keep at least the threshold of
                // ballots. Otherwise their keep factor was too low, i.e. too
                // much of their ballots were unduly transfered to other
                // candidates.
                Status::Elected => {
                    if log_enabled!(Level::Warn) && count.sum[i] < self.threshold {
                        warn!(
                            "Count for elected candidate #{i} ({}) is lower than the quota: {} < {} ~ {} < {}",
                            candidate.nickname,
                            count.sum[i],
                            self.threshold,
                            count.sum[i].to_f64(),
                            self.threshold.to_f64(),
                        )
                    }
                }
                // Already non-elected candidates must have no ballot left.
                Status::NotElected => assert!(count.sum[i].is_zero()),
                Status::Candidate => {
                    // Elect candidates that exceeded the threshold.
                    let exceeds_threshold = if R::is_exact() {
                        count.sum[i] > self.threshold
                    } else {
                        count.sum[i] >= self.threshold
                    };
                    if exceeds_threshold {
                        // We cannot elect more candidates than seats.
                        assert!(self.to_elect > 0);
                        self.elect_candidate(i, ElectedReason::ExceedsQuota);
                        self.write_action(
                            stdout,
                            &format!("Elect: {}", candidate.name),
                            count,
                            true,
                        )?;
                        has_elected = true;
                    }
                }
                Status::Withdrawn => (),
            }
        }
        Ok(has_elected)
    }

    /// Elects or defeat all remaining candidates, once a quorum is defeated or
    /// elected (respectively).
    fn handle_remaining_candidates(
        &mut self,
        stdout: &mut impl io::Write,
        count: &mut VoteCount<I, R>,
    ) -> io::Result<()> {
        debug!("Checking remaining candidates");
        for (i, candidate) in self.election.candidates.iter().enumerate() {
            if self.statuses[i] == Status::Candidate {
                if self.elected.len() < self.election.num_seats {
                    self.elect_candidate(i, ElectedReason::Remaining);
                    self.write_action(
                        stdout,
                        &format!("Elect remaining: {}", candidate.name),
                        count,
                        true,
                    )?;
                } else {
                    self.defeat_candidate(i, NotElectedReason::Remaining);
                    self.write_action(
                        stdout,
                        &format!("Defeat remaining: {}", candidate.name),
                        count,
                        true,
                    )?;
                    self.keep_factors[i] = R::zero();
                    count.sum[i] = R::zero();
                }
                *count = self.count_votes();
            }
        }
        Ok(())
    }

    /// Returns the next candidate to defeat.
    fn next_defeated_candidate(
        &mut self,
        stdout: &mut impl io::Write,
        count: &VoteCount<I, R>,
    ) -> io::Result<usize> {
        let min_sum = count
            .sum
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                if self.statuses[i] == Status::Candidate {
                    Some(x)
                } else {
                    None
                }
            })
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        debug!("Lowest vote: {min_sum} ~ {}", min_sum.to_f64());

        let low_threshold = min_sum + &self.surplus;
        debug!(
            "Low threshold: {low_threshold} ~ {}",
            low_threshold.to_f64()
        );

        let low_candidates = (0..self.num_candidates())
            .filter(|&i| self.statuses[i] == Status::Candidate && count.sum[i] <= low_threshold)
            .collect::<Vec<_>>();
        debug!("Low candidates: {low_candidates:?}");

        if low_candidates.len() == 1 {
            return Ok(low_candidates[0]);
        }

        let defeated = *low_candidates
            .iter()
            .min_by_key(|c| self.election.tie_order.get(c).unwrap())
            .unwrap();
        self.transcript.push(TranscriptEvent::BreakTie {
            list: low_candidates.clone(),
            chosen: defeated,
        });

        let low_candidates_list: String = low_candidates
            .into_iter()
            .map(|c| &self.election.candidates[c].name)
            .fold(String::new(), |mut s, c| {
                if !s.is_empty() {
                    s.push_str(", ");
                }
                s.push_str(c);
                s
            });
        self.write_action(
            stdout,
            &format!(
                "Break tie (defeat): [{}] -> {}",
                low_candidates_list, self.election.candidates[defeated].name
            ),
            count,
            false,
        )?;

        Ok(defeated)
    }

    /// Writes an action to the given output.
    fn write_action(
        &self,
        stdout: &mut impl io::Write,
        action: &str,
        count: &VoteCount<I, R>,
        print_full_count: bool,
    ) -> io::Result<()> {
        writeln!(stdout, "Action: {action}")?;
        if print_full_count {
            self.write_candidate_counts(stdout, count)?;
        }
        count.write_stats(stdout, &self.threshold, &self.surplus)?;
        Ok(())
    }

    /// Writes current candidate counts to the given output.
    fn write_candidate_counts(
        &self,
        stdout: &mut impl io::Write,
        count: &VoteCount<I, R>,
    ) -> io::Result<()> {
        // Sort candidates: elected first, then candidates and lastly
        // non-elected.
        let mut droop_sorted_candidates: Vec<usize> = (0..self.num_candidates()).collect();
        droop_sorted_candidates.sort_by_key(|&i| match self.statuses[i] {
            Status::Withdrawn | Status::Elected => 0,
            Status::Candidate => 1,
            Status::NotElected => {
                if !count.sum[i].is_zero() {
                    2
                } else {
                    3
                }
            }
        });

        // Find the first defeated candidate to have a zero count (if any).
        let split = droop_sorted_candidates
            .iter()
            .position(|&i| self.statuses[i] == Status::NotElected && count.sum[i] == R::zero());

        // Simply print the candidates with a non-zero count.
        for &i in droop_sorted_candidates[..split.unwrap_or(droop_sorted_candidates.len())].iter() {
            let status = match self.statuses[i] {
                Status::Elected => "Elected: ",
                Status::Candidate => "Hopeful: ",
                Status::NotElected => "Defeated:",
                Status::Withdrawn => continue,
            };
            writeln!(
                stdout,
                "\t{status} {} ({})",
                self.election.candidates[i].name, count.sum[i]
            )?;
        }

        // Candidates beyond the first defeated one with a zero count must all
        // be defeated and with a zero count.
        if let Some(split) = split {
            write!(stdout, "\tDefeated: ")?;
            for (rank, &i) in droop_sorted_candidates[split..].iter().enumerate() {
                assert_eq!(self.statuses[i], Status::NotElected);
                assert_eq!(count.sum[i], R::zero());
                if rank != 0 {
                    write!(stdout, ", ")?;
                }
                write!(stdout, "{}", self.election.candidates[i].name)?;
            }
            writeln!(stdout, " ({})", R::zero())?;
        }

        Ok(())
    }

    /// Displays a debug output of the current vote counts.
    fn debug_count(&self, count: &VoteCount<I, R>) {
        if !log_enabled!(Level::Debug) {
            return;
        }

        let mut sorted_candidates: Vec<usize> = (0..self.num_candidates()).collect();
        sorted_candidates.sort_by(|&i, &j| count.sum[i].partial_cmp(&count.sum[j]).unwrap());

        debug!("Sums:");
        for (rank, &i) in sorted_candidates.iter().rev().enumerate() {
            debug!(
                "    [{rank}] {} ({:?}) ~ {}",
                self.election.candidates[i].nickname,
                self.statuses[i],
                count.sum[i].to_f64()
            );
        }
        debug!("    @exhausted ~ {}", count.exhausted.to_f64());

        let sum = count.sum.iter().sum::<R>();
        debug!("Sum = {sum}");
        let total = sum + &count.exhausted;
        debug!("Total = {total}");
        R::assert_eq(
            total,
            R::from_usize(self.election.num_ballots),
            "Total count must be equal to the number of ballots",
        );

        debug!("Elected:");
        for (i, &id) in self.elected.iter().enumerate() {
            debug!("    [{i}] {}", self.election.candidates[id].nickname);
        }
        debug!("Not elected:");
        for (i, &id) in self.not_elected.iter().enumerate() {
            debug!(
                "    [{}] {}",
                self.num_candidates() - i - 1,
                self.election.candidates[id].nickname
            );
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::arithmetic::{FixedDecimal9, Integer64};
    use crate::types::TranscriptEvent::*;
    use crate::types::{Ballot, Candidate};
    use crate::util::log_tester::ThreadLocalLogger;
    use log::Level::{Debug, Info, Warn};
    use num::BigRational;
    use num::traits::{One, Zero};

    pub struct StateBuilder<'e, I, R> {
        election: Option<&'e Election>,
        statuses: Option<Vec<Status>>,
        keep_factors: Option<Vec<R>>,
        threshold: Option<R>,
        surplus: Option<R>,
        omega: Option<R>,
        parallel: Option<Parallel>,
        force_positive_surplus: Option<bool>,
        pascal: Option<Option<&'e [Vec<I>]>>,
        _phantom: PhantomData<I>,
    }

    impl<I, R> Default for StateBuilder<'_, I, R> {
        fn default() -> Self {
            StateBuilder {
                election: None,
                statuses: None,
                keep_factors: None,
                threshold: None,
                surplus: None,
                omega: None,
                parallel: None,
                force_positive_surplus: None,
                pascal: None,
                _phantom: PhantomData,
            }
        }
    }

    impl<'e, I, R> StateBuilder<'e, I, R>
    where
        R: Clone,
    {
        fn election(mut self, election: &'e Election) -> Self {
            self.election = Some(election);
            self
        }

        fn statuses(mut self, statuses: impl Into<Vec<Status>>) -> Self {
            self.statuses = Some(statuses.into());
            self
        }

        fn keep_factors(mut self, keep_factors: impl Into<Vec<R>>) -> Self {
            self.keep_factors = Some(keep_factors.into());
            self
        }

        fn threshold(mut self, threshold: R) -> Self {
            self.threshold = Some(threshold);
            self
        }

        fn surplus(mut self, surplus: R) -> Self {
            self.surplus = Some(surplus);
            self
        }

        fn omega(mut self, omega: R) -> Self {
            self.omega = Some(omega);
            self
        }

        fn parallel(mut self, parallel: Parallel) -> Self {
            self.parallel = Some(parallel);
            self
        }

        fn force_positive_surplus(mut self, force_positive_surplus: bool) -> Self {
            self.force_positive_surplus = Some(force_positive_surplus);
            self
        }

        fn pascal(mut self, pascal: Option<&'e [Vec<I>]>) -> Self {
            self.pascal = Some(pascal);
            self
        }

        #[track_caller]
        fn build(self) -> State<'static, 'e, I, R> {
            let election = self.election.unwrap();
            let statuses = self.statuses.unwrap();
            let elected = statuses
                .iter()
                .enumerate()
                .filter_map(|(i, status)| {
                    if let Status::Elected = status {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let not_elected = statuses
                .iter()
                .enumerate()
                .filter_map(|(i, status)| {
                    if let Status::NotElected = status {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let to_elect = election.num_seats - elected.len();
            let keep_factors = self.keep_factors.unwrap();

            let mut transcript = Vec::new();
            for (i, status) in statuses.iter().enumerate() {
                match status {
                    Status::Candidate => (),
                    Status::Withdrawn => transcript.push(TranscriptEvent::Withdrawn(i)),
                    Status::Elected => {
                        transcript.push(TranscriptEvent::Elected(i, ElectedReason::Other))
                    }
                    Status::NotElected => {
                        transcript.push(TranscriptEvent::NotElected(i, NotElectedReason::Other))
                    }
                }
            }
            transcript.push(TranscriptEvent::KeepFactors(keep_factors.clone()));

            State {
                election,
                transcript,
                statuses,
                elected,
                not_elected,
                to_elect,
                keep_factors,
                threshold: self.threshold.unwrap(),
                surplus: self.surplus.unwrap(),
                omega: self.omega.unwrap(),
                parallel: self.parallel.unwrap(),
                force_positive_surplus: self.force_positive_surplus.unwrap(),
                pascal: self.pascal.unwrap(),
                thread_pool: None,
                _phantom: PhantomData,
            }
        }
    }

    fn make_fake_election() -> Election {
        Election::builder()
            .title("Vegetable contest")
            .num_seats(3)
            .candidates([
                Candidate::new("apple", false),
                Candidate::new("banana", true),
                Candidate::new("cherry", false),
                Candidate::new("date", false),
                Candidate::new("eggplant", false),
                Candidate::new("fig", true),
                Candidate::new("grape", false),
                Candidate::new("hazelnut", false),
                Candidate::new("jalapeno", false),
            ])
            .num_ballots(6)
            .build()
    }

    fn make_fake_state(election: &Election) -> State<'_, '_, Integer64, FixedDecimal9> {
        State::builder()
            .election(election)
            .statuses([
                Status::Candidate,
                Status::Withdrawn,
                Status::Elected,
                Status::NotElected,
                Status::Candidate,
                Status::NotElected,
                Status::Elected,
                Status::NotElected,
                Status::NotElected,
            ])
            .keep_factors([
                FixedDecimal9::one(),
                FixedDecimal9::zero(),
                FixedDecimal9::ratio(1, 2),
                FixedDecimal9::zero(),
                FixedDecimal9::one(),
                FixedDecimal9::zero(),
                FixedDecimal9::ratio(2, 3),
                FixedDecimal9::zero(),
                FixedDecimal9::zero(),
            ])
            .threshold(FixedDecimal9::ratio(3, 2))
            .surplus(FixedDecimal9::ratio(1, 10))
            .omega(FixedDecimal9::ratio(1, 1_000_000))
            .parallel(Parallel::No)
            .force_positive_surplus(false)
            .pascal(None)
            .build()
    }

    fn make_fake_count() -> VoteCount<Integer64, FixedDecimal9> {
        VoteCount::new(
            [
                FixedDecimal9::ratio(7, 9),
                FixedDecimal9::zero(),
                FixedDecimal9::ratio(5, 3),
                FixedDecimal9::ratio(1, 10),
                FixedDecimal9::ratio(7, 8),
                FixedDecimal9::zero(),
                FixedDecimal9::ratio(11, 6),
                FixedDecimal9::ratio(2, 10),
                FixedDecimal9::zero(),
            ],
            FixedDecimal9::new(547222224),
        )
    }

    #[test]
    fn test_stv_droop_parallel_no() {
        test_stv_droop(Parallel::No);
    }

    #[test]
    fn test_stv_droop_parallel_rayon() {
        test_stv_droop(Parallel::Rayon);
    }

    #[test]
    fn test_stv_droop_parallel_custom() {
        test_stv_droop(Parallel::Custom);
    }

    fn test_stv_droop(parallel: Parallel) {
        let election = Election::builder()
            .title("Vegetable contest")
            .num_seats(5)
            .candidates([
                Candidate::new("apple", false),
                Candidate::new("banana", true),
                Candidate::new("cherry", false),
                Candidate::new("date", false),
                Candidate::new("eggplant", false),
                Candidate::new("fig", true),
                Candidate::new("grape", false),
                Candidate::new("hazelnut", false),
                Candidate::new("jalapeno", false),
            ])
            .ballots([
                Ballot::new(1, [vec![0]]),
                Ballot::new(2, [vec![2]]),
                Ballot::new(3, [vec![3]]),
                Ballot::new(4, [vec![4]]),
                Ballot::new(5, [vec![6]]),
                Ballot::new(6, [vec![7]]),
                Ballot::new(7, [vec![8]]),
            ])
            .build();

        let mut buf = Vec::new();
        let result = stv_droop::<Integer64, FixedDecimal9>(
            &mut buf,
            &election,
            "package name",
            6,
            parallel,
            None,
            false,
            false,
            false,
        )
        .unwrap();

        assert_eq!(
            result,
            ElectionResult {
                elected: vec![6, 7, 8, 4, 3],
                not_elected: vec![0, 2],
                transcript: vec![
                    Withdrawn(1),
                    Withdrawn(5),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(4000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(5000000000),
                            FixedDecimal9::new(6000000000),
                            FixedDecimal9::new(7000000000)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    StartRound(1),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(4000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(5000000000),
                            FixedDecimal9::new(6000000000),
                            FixedDecimal9::new(7000000000)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    Quota(FixedDecimal9::new(4666666667)),
                    Elected(6, ElectedReason::ExceedsQuota),
                    Elected(7, ElectedReason::ExceedsQuota),
                    Elected(8, ElectedReason::ExceedsQuota),
                    Surplus(FixedDecimal9::new(3999999999)),
                    StartRound(2),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(4000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(5000000000),
                            FixedDecimal9::new(6000000000),
                            FixedDecimal9::new(7000000000)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    Quota(FixedDecimal9::new(4666666667)),
                    Surplus(FixedDecimal9::new(3999999999)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(933333334),
                        FixedDecimal9::new(777777778),
                        FixedDecimal9::new(666666667)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(4000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(4666666670),
                            FixedDecimal9::new(4666666668),
                            FixedDecimal9::new(4666666669)
                        ],
                        exhausted: FixedDecimal9::new(3999999993)
                    },
                    Quota(FixedDecimal9::new(4000000002)),
                    Surplus(FixedDecimal9::new(2000000001)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(800000001),
                        FixedDecimal9::new(666666668),
                        FixedDecimal9::new(571428572)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(4000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(4000000005),
                            FixedDecimal9::new(4000000008),
                            FixedDecimal9::new(4000000004)
                        ],
                        exhausted: FixedDecimal9::new(5999999983)
                    },
                    Quota(FixedDecimal9::new(3666666670)),
                    Elected(4, ElectedReason::ExceedsQuota),
                    Surplus(FixedDecimal9::new(1333333337)),
                    StartRound(3),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(4000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(4000000005),
                            FixedDecimal9::new(4000000008),
                            FixedDecimal9::new(4000000004)
                        ],
                        exhausted: FixedDecimal9::new(5999999983)
                    },
                    Quota(FixedDecimal9::new(3666666670)),
                    Surplus(FixedDecimal9::new(1333333337)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(916666668),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(733333335),
                        FixedDecimal9::new(611111112),
                        FixedDecimal9::new(523809525)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3666666672),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3666666675),
                            FixedDecimal9::new(3666666672),
                            FixedDecimal9::new(3666666675)
                        ],
                        exhausted: FixedDecimal9::new(7333333306)
                    },
                    Quota(FixedDecimal9::new(3444444450)),
                    Surplus(FixedDecimal9::new(888888894)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(861111113),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(688888891),
                        FixedDecimal9::new(574074076),
                        FixedDecimal9::new(492063494)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3444444452),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3444444455),
                            FixedDecimal9::new(3444444456),
                            FixedDecimal9::new(3444444458)
                        ],
                        exhausted: FixedDecimal9::new(8222222179)
                    },
                    Quota(FixedDecimal9::new(3296296304)),
                    Surplus(FixedDecimal9::new(592592605)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(824074077),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(659259262),
                        FixedDecimal9::new(549382718),
                        FixedDecimal9::new(470899473)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3296296308),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3296296310),
                            FixedDecimal9::new(3296296308),
                            FixedDecimal9::new(3296296311)
                        ],
                        exhausted: FixedDecimal9::new(8814814763)
                    },
                    Quota(FixedDecimal9::new(3197530873)),
                    Surplus(FixedDecimal9::new(395061745)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(799382719),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(639506175),
                        FixedDecimal9::new(532921813),
                        FixedDecimal9::new(456790125)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3197530876),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3197530875),
                            FixedDecimal9::new(3197530878),
                            FixedDecimal9::new(3197530875)
                        ],
                        exhausted: FixedDecimal9::new(9209876496)
                    },
                    Quota(FixedDecimal9::new(3131687251)),
                    Surplus(FixedDecimal9::new(263374500)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(782921813),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(626337451),
                        FixedDecimal9::new(521947876),
                        FixedDecimal9::new(447383894)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3131687252),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3131687255),
                            FixedDecimal9::new(3131687256),
                            FixedDecimal9::new(3131687258)
                        ],
                        exhausted: FixedDecimal9::new(9473250979)
                    },
                    Quota(FixedDecimal9::new(3087791504)),
                    Surplus(FixedDecimal9::new(175583005)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(771947877),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(617558301),
                        FixedDecimal9::new(514631918),
                        FixedDecimal9::new(441113073)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3087791508),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3087791505),
                            FixedDecimal9::new(3087791508),
                            FixedDecimal9::new(3087791511)
                        ],
                        exhausted: FixedDecimal9::new(9648833968)
                    },
                    Quota(FixedDecimal9::new(3058527673)),
                    Surplus(FixedDecimal9::new(117055340)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(764631919),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(611705535),
                        FixedDecimal9::new(509754613),
                        FixedDecimal9::new(436932525)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3058527676),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3058527675),
                            FixedDecimal9::new(3058527678),
                            FixedDecimal9::new(3058527675)
                        ],
                        exhausted: FixedDecimal9::new(9765889296)
                    },
                    Quota(FixedDecimal9::new(3039018451)),
                    Surplus(FixedDecimal9::new(78036900)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(759754614),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(607803691),
                        FixedDecimal9::new(506503076),
                        FixedDecimal9::new(434145494)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3039018456),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3039018455),
                            FixedDecimal9::new(3039018456),
                            FixedDecimal9::new(3039018458)
                        ],
                        exhausted: FixedDecimal9::new(9843926175)
                    },
                    Quota(FixedDecimal9::new(3026012305)),
                    Surplus(FixedDecimal9::new(52024605)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(756503077),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(605202462),
                        FixedDecimal9::new(504335385),
                        FixedDecimal9::new(432287473)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3026012308),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3026012310),
                            FixedDecimal9::new(3026012310),
                            FixedDecimal9::new(3026012311)
                        ],
                        exhausted: FixedDecimal9::new(9895950761)
                    },
                    Quota(FixedDecimal9::new(3017341540)),
                    Surplus(FixedDecimal9::new(34683079)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(754335386),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(603468309),
                        FixedDecimal9::new(502890257),
                        FixedDecimal9::new(431048792)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3017341544),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3017341545),
                            FixedDecimal9::new(3017341542),
                            FixedDecimal9::new(3017341544)
                        ],
                        exhausted: FixedDecimal9::new(9930633825)
                    },
                    Quota(FixedDecimal9::new(3011561030)),
                    Surplus(FixedDecimal9::new(23122055)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(752890258),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(602312207),
                        FixedDecimal9::new(501926839),
                        FixedDecimal9::new(430223005)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3011561032),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3011561035),
                            FixedDecimal9::new(3011561034),
                            FixedDecimal9::new(3011561035)
                        ],
                        exhausted: FixedDecimal9::new(9953755864)
                    },
                    Quota(FixedDecimal9::new(3007707357)),
                    Surplus(FixedDecimal9::new(15414708)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(751926840),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(601541472),
                        FixedDecimal9::new(501284560),
                        FixedDecimal9::new(429672480)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3007707360),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3007707360),
                            FixedDecimal9::new(3007707360),
                            FixedDecimal9::new(3007707360)
                        ],
                        exhausted: FixedDecimal9::new(9969170560)
                    },
                    Quota(FixedDecimal9::new(3005138241)),
                    Surplus(FixedDecimal9::new(10276476)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(751284561),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(601027649),
                        FixedDecimal9::new(500856374),
                        FixedDecimal9::new(429305464)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3005138244),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3005138245),
                            FixedDecimal9::new(3005138244),
                            FixedDecimal9::new(3005138248)
                        ],
                        exhausted: FixedDecimal9::new(9979447019)
                    },
                    Quota(FixedDecimal9::new(3003425497)),
                    Surplus(FixedDecimal9::new(6850993)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750856375),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600685100),
                        FixedDecimal9::new(500570917),
                        FixedDecimal9::new(429060786)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3003425500),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3003425500),
                            FixedDecimal9::new(3003425502),
                            FixedDecimal9::new(3003425502)
                        ],
                        exhausted: FixedDecimal9::new(9986297996)
                    },
                    Quota(FixedDecimal9::new(3002283668)),
                    Surplus(FixedDecimal9::new(4567332)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750570918),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600456734),
                        FixedDecimal9::new(500380612),
                        FixedDecimal9::new(428897668)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3002283672),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3002283670),
                            FixedDecimal9::new(3002283672),
                            FixedDecimal9::new(3002283676)
                        ],
                        exhausted: FixedDecimal9::new(9990865310)
                    },
                    Quota(FixedDecimal9::new(3001522449)),
                    Surplus(FixedDecimal9::new(3044894)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750380613),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600304490),
                        FixedDecimal9::new(500253742),
                        FixedDecimal9::new(428788922)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3001522452),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3001522450),
                            FixedDecimal9::new(3001522452),
                            FixedDecimal9::new(3001522454)
                        ],
                        exhausted: FixedDecimal9::new(9993910192)
                    },
                    Quota(FixedDecimal9::new(3001014969)),
                    Surplus(FixedDecimal9::new(2029932)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750253743),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600202994),
                        FixedDecimal9::new(500169162),
                        FixedDecimal9::new(428716425)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3001014972),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3001014970),
                            FixedDecimal9::new(3001014972),
                            FixedDecimal9::new(3001014975)
                        ],
                        exhausted: FixedDecimal9::new(9995940111)
                    },
                    Quota(FixedDecimal9::new(3000676649)),
                    Surplus(FixedDecimal9::new(1353293)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750169163),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600135330),
                        FixedDecimal9::new(500112775),
                        FixedDecimal9::new(428668093)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000676652),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000676650),
                            FixedDecimal9::new(3000676650),
                            FixedDecimal9::new(3000676651)
                        ],
                        exhausted: FixedDecimal9::new(9997293397)
                    },
                    Quota(FixedDecimal9::new(3000451101)),
                    Surplus(FixedDecimal9::new(902199)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750112776),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600090221),
                        FixedDecimal9::new(500075184),
                        FixedDecimal9::new(428635872)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000451104),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000451105),
                            FixedDecimal9::new(3000451104),
                            FixedDecimal9::new(3000451104)
                        ],
                        exhausted: FixedDecimal9::new(9998195583)
                    },
                    Quota(FixedDecimal9::new(3000300737)),
                    Surplus(FixedDecimal9::new(601469)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750075185),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600060148),
                        FixedDecimal9::new(500050124),
                        FixedDecimal9::new(428614392)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000300740),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000300740),
                            FixedDecimal9::new(3000300744),
                            FixedDecimal9::new(3000300744)
                        ],
                        exhausted: FixedDecimal9::new(9998797032)
                    },
                    Quota(FixedDecimal9::new(3000200495)),
                    Surplus(FixedDecimal9::new(400988)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750050124),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600040100),
                        FixedDecimal9::new(500033416),
                        FixedDecimal9::new(428600072)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000200496),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000200500),
                            FixedDecimal9::new(3000200496),
                            FixedDecimal9::new(3000200504)
                        ],
                        exhausted: FixedDecimal9::new(9999198004)
                    },
                    Quota(FixedDecimal9::new(3000133667)),
                    Surplus(FixedDecimal9::new(267328)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750033417),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600026734),
                        FixedDecimal9::new(500022278),
                        FixedDecimal9::new(428590524)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000133668),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000133670),
                            FixedDecimal9::new(3000133668),
                            FixedDecimal9::new(3000133668)
                        ],
                        exhausted: FixedDecimal9::new(9999465326)
                    },
                    Quota(FixedDecimal9::new(3000089113)),
                    Surplus(FixedDecimal9::new(178222)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750022279),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600017823),
                        FixedDecimal9::new(500014853),
                        FixedDecimal9::new(428584160)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000089116),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000089115),
                            FixedDecimal9::new(3000089118),
                            FixedDecimal9::new(3000089120)
                        ],
                        exhausted: FixedDecimal9::new(9999643531)
                    },
                    Quota(FixedDecimal9::new(3000059412)),
                    Surplus(FixedDecimal9::new(118821)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750014854),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600011883),
                        FixedDecimal9::new(500009903),
                        FixedDecimal9::new(428579917)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000059416),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000059415),
                            FixedDecimal9::new(3000059418),
                            FixedDecimal9::new(3000059419)
                        ],
                        exhausted: FixedDecimal9::new(9999762332)
                    },
                    Quota(FixedDecimal9::new(3000039612)),
                    Surplus(FixedDecimal9::new(79220)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750009904),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600007923),
                        FixedDecimal9::new(500006603),
                        FixedDecimal9::new(428577088)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000039616),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000039615),
                            FixedDecimal9::new(3000039618),
                            FixedDecimal9::new(3000039616)
                        ],
                        exhausted: FixedDecimal9::new(9999841535)
                    },
                    Quota(FixedDecimal9::new(3000026411)),
                    Surplus(FixedDecimal9::new(52821)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750006603),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600005283),
                        FixedDecimal9::new(500004402),
                        FixedDecimal9::new(428575202)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000026412),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000026415),
                            FixedDecimal9::new(3000026412),
                            FixedDecimal9::new(3000026414)
                        ],
                        exhausted: FixedDecimal9::new(9999894347)
                    },
                    Quota(FixedDecimal9::new(3000017609)),
                    Surplus(FixedDecimal9::new(35217)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750004403),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600003522),
                        FixedDecimal9::new(500002935),
                        FixedDecimal9::new(428573945)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000017612),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000017610),
                            FixedDecimal9::new(3000017610),
                            FixedDecimal9::new(3000017615)
                        ],
                        exhausted: FixedDecimal9::new(9999929553)
                    },
                    Quota(FixedDecimal9::new(3000011742)),
                    Surplus(FixedDecimal9::new(23479)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750002936),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600002349),
                        FixedDecimal9::new(500001958),
                        FixedDecimal9::new(428573107)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000011744),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000011745),
                            FixedDecimal9::new(3000011748),
                            FixedDecimal9::new(3000011749)
                        ],
                        exhausted: FixedDecimal9::new(9999953014)
                    },
                    Quota(FixedDecimal9::new(3000007832)),
                    Surplus(FixedDecimal9::new(15658)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750001959),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600001567),
                        FixedDecimal9::new(500001306),
                        FixedDecimal9::new(428572548)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000007836),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000007835),
                            FixedDecimal9::new(3000007836),
                            FixedDecimal9::new(3000007836)
                        ],
                        exhausted: FixedDecimal9::new(9999968657)
                    },
                    Quota(FixedDecimal9::new(3000005224)),
                    Surplus(FixedDecimal9::new(10447)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750001307),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600001045),
                        FixedDecimal9::new(500000871),
                        FixedDecimal9::new(428572175)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000005228),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000005225),
                            FixedDecimal9::new(3000005226),
                            FixedDecimal9::new(3000005225)
                        ],
                        exhausted: FixedDecimal9::new(9999979096)
                    },
                    Quota(FixedDecimal9::new(3000003485)),
                    Surplus(FixedDecimal9::new(6964)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750000872),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600000698),
                        FixedDecimal9::new(500000581),
                        FixedDecimal9::new(428571927)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000003488),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000003490),
                            FixedDecimal9::new(3000003486),
                            FixedDecimal9::new(3000003489)
                        ],
                        exhausted: FixedDecimal9::new(9999986047)
                    },
                    Quota(FixedDecimal9::new(3000002326)),
                    Surplus(FixedDecimal9::new(4649)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750000582),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600000466),
                        FixedDecimal9::new(500000388),
                        FixedDecimal9::new(428571761)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000002328),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000002330),
                            FixedDecimal9::new(3000002328),
                            FixedDecimal9::new(3000002327)
                        ],
                        exhausted: FixedDecimal9::new(9999990687)
                    },
                    Quota(FixedDecimal9::new(3000001553)),
                    Surplus(FixedDecimal9::new(3101)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750000389),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600000311),
                        FixedDecimal9::new(500000259),
                        FixedDecimal9::new(428571651)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000001556),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000001555),
                            FixedDecimal9::new(3000001554),
                            FixedDecimal9::new(3000001557)
                        ],
                        exhausted: FixedDecimal9::new(9999993778)
                    },
                    Quota(FixedDecimal9::new(3000001038)),
                    Surplus(FixedDecimal9::new(2070)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750000260),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600000208),
                        FixedDecimal9::new(500000174),
                        FixedDecimal9::new(428571577)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000001040),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000001040),
                            FixedDecimal9::new(3000001044),
                            FixedDecimal9::new(3000001039)
                        ],
                        exhausted: FixedDecimal9::new(9999995837)
                    },
                    Quota(FixedDecimal9::new(3000000694)),
                    Surplus(FixedDecimal9::new(1387)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750000174),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600000139),
                        FixedDecimal9::new(500000116),
                        FixedDecimal9::new(428571528)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1000000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000000696),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000000695),
                            FixedDecimal9::new(3000000696),
                            FixedDecimal9::new(3000000696)
                        ],
                        exhausted: FixedDecimal9::new(9999997217)
                    },
                    Quota(FixedDecimal9::new(3000000464)),
                    Surplus(FixedDecimal9::new(927)),
                    NotElected(0, NotElectedReason::LowSurplus),
                    KeepFactors(vec![
                        FixedDecimal9::new(0),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(750000174),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(600000139),
                        FixedDecimal9::new(500000116),
                        FixedDecimal9::new(428571528)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000000696),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000000695),
                            FixedDecimal9::new(3000000696),
                            FixedDecimal9::new(3000000696)
                        ],
                        exhausted: FixedDecimal9::new(10999997217)
                    },
                    StartRound(4),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(2000000000),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000000696),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000000695),
                            FixedDecimal9::new(3000000696),
                            FixedDecimal9::new(3000000696)
                        ],
                        exhausted: FixedDecimal9::new(10999997217)
                    },
                    Quota(FixedDecimal9::new(2833333798)),
                    Elected(3, ElectedReason::ExceedsQuota),
                    Surplus(FixedDecimal9::new(833333793)),
                    NotElected(2, NotElectedReason::Remaining),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(3000000696),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3000000695),
                            FixedDecimal9::new(3000000696),
                            FixedDecimal9::new(3000000696)
                        ],
                        exhausted: FixedDecimal9::new(12999997217)
                    }
                ],
            },
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	package name
	Rule: Meek Parametric (omega = 1/10^6)
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 5
	Ballots: 28
	Quota: 4.666666667
	Omega: 0.000001000

	Add eligible: Apple
	Add withdrawn: Banana
	Add eligible: Cherry
	Add eligible: Date
	Add eligible: Eggplant
	Add withdrawn: Fig
	Add eligible: Grape
	Add eligible: Hazelnut
	Add eligible: Jalapeno
Action: Begin Count
	Hopeful:  Apple (1.000000000)
	Hopeful:  Cherry (2.000000000)
	Hopeful:  Date (3.000000000)
	Hopeful:  Eggplant (4.000000000)
	Hopeful:  Grape (5.000000000)
	Hopeful:  Hazelnut (6.000000000)
	Hopeful:  Jalapeno (7.000000000)
	Quota: 4.666666667
	Votes: 28.000000000
	Residual: 0.000000000
	Total: 28.000000000
	Surplus: 0.000000000
Round 1:
Action: Elect: Grape
	Elected:  Grape (5.000000000)
	Hopeful:  Apple (1.000000000)
	Hopeful:  Cherry (2.000000000)
	Hopeful:  Date (3.000000000)
	Hopeful:  Eggplant (4.000000000)
	Hopeful:  Hazelnut (6.000000000)
	Hopeful:  Jalapeno (7.000000000)
	Quota: 4.666666667
	Votes: 28.000000000
	Residual: 0.000000000
	Total: 28.000000000
	Surplus: 0.000000000
Action: Elect: Hazelnut
	Elected:  Grape (5.000000000)
	Elected:  Hazelnut (6.000000000)
	Hopeful:  Apple (1.000000000)
	Hopeful:  Cherry (2.000000000)
	Hopeful:  Date (3.000000000)
	Hopeful:  Eggplant (4.000000000)
	Hopeful:  Jalapeno (7.000000000)
	Quota: 4.666666667
	Votes: 28.000000000
	Residual: 0.000000000
	Total: 28.000000000
	Surplus: 0.000000000
Action: Elect: Jalapeno
	Elected:  Grape (5.000000000)
	Elected:  Hazelnut (6.000000000)
	Elected:  Jalapeno (7.000000000)
	Hopeful:  Apple (1.000000000)
	Hopeful:  Cherry (2.000000000)
	Hopeful:  Date (3.000000000)
	Hopeful:  Eggplant (4.000000000)
	Quota: 4.666666667
	Votes: 28.000000000
	Residual: 0.000000000
	Total: 28.000000000
	Surplus: 0.000000000
Action: Iterate (elected)
	Quota: 4.666666667
	Votes: 28.000000000
	Residual: 0.000000000
	Total: 28.000000000
	Surplus: 3.999999999
Round 2:
Action: Elect: Eggplant
	Elected:  Eggplant (4.000000000)
	Elected:  Grape (4.000000005)
	Elected:  Hazelnut (4.000000008)
	Elected:  Jalapeno (4.000000004)
	Hopeful:  Apple (1.000000000)
	Hopeful:  Cherry (2.000000000)
	Hopeful:  Date (3.000000000)
	Quota: 3.666666670
	Votes: 22.000000017
	Residual: 5.999999983
	Total: 28.000000000
	Surplus: 2.000000001
Action: Iterate (elected)
	Quota: 3.666666670
	Votes: 22.000000017
	Residual: 5.999999983
	Total: 28.000000000
	Surplus: 1.333333337
Round 3:
Action: Iterate (omega)
	Quota: 3.000000464
	Votes: 18.000002783
	Residual: 9.999997217
	Total: 28.000000000
	Surplus: 0.000000927
Action: Defeat (surplus 0.000000927 < omega): Apple
	Elected:  Eggplant (3.000000696)
	Elected:  Grape (3.000000695)
	Elected:  Hazelnut (3.000000696)
	Elected:  Jalapeno (3.000000696)
	Hopeful:  Cherry (2.000000000)
	Hopeful:  Date (3.000000000)
	Defeated: Apple (1.000000000)
	Quota: 3.000000464
	Votes: 18.000002783
	Residual: 9.999997217
	Total: 28.000000000
	Surplus: 0.000000927
Round 4:
Action: Elect: Date
	Elected:  Date (3.000000000)
	Elected:  Eggplant (3.000000696)
	Elected:  Grape (3.000000695)
	Elected:  Hazelnut (3.000000696)
	Elected:  Jalapeno (3.000000696)
	Hopeful:  Cherry (2.000000000)
	Defeated: Apple (0.000000000)
	Quota: 2.833333798
	Votes: 17.000002783
	Residual: 10.999997217
	Total: 28.000000000
	Surplus: 0.000000927
Action: Iterate (elected)
	Quota: 2.833333798
	Votes: 17.000002783
	Residual: 10.999997217
	Total: 28.000000000
	Surplus: 0.833333793
Action: Defeat remaining: Cherry
	Elected:  Date (3.000000000)
	Elected:  Eggplant (3.000000696)
	Elected:  Grape (3.000000695)
	Elected:  Hazelnut (3.000000696)
	Elected:  Jalapeno (3.000000696)
	Defeated: Cherry (2.000000000)
	Defeated: Apple (0.000000000)
	Quota: 2.833333798
	Votes: 17.000002783
	Residual: 10.999997217
	Total: 28.000000000
	Surplus: 0.833333793
Action: Count Complete
	Elected:  Date (3.000000000)
	Elected:  Eggplant (3.000000696)
	Elected:  Grape (3.000000695)
	Elected:  Hazelnut (3.000000696)
	Elected:  Jalapeno (3.000000696)
	Defeated: Apple, Cherry (0.000000000)
	Quota: 2.833333798
	Votes: 15.000002783
	Residual: 12.999997217
	Total: 28.000000000
	Surplus: 0.833333793

"
        );
    }

    #[test]
    fn test_stv_droop_equalize_parallel_no() {
        test_stv_droop_equalize(Parallel::No);
    }

    #[test]
    fn test_stv_droop_equalize_parallel_rayon() {
        test_stv_droop_equalize(Parallel::Rayon);
    }

    #[test]
    fn test_stv_droop_equalize_parallel_custom() {
        test_stv_droop_equalize(Parallel::Custom);
    }

    fn test_stv_droop_equalize(parallel: Parallel) {
        let election = Election::builder()
            .title("Vegetable contest")
            .num_seats(3)
            .candidates([
                Candidate::new("apple", false),
                Candidate::new("banana", false),
                Candidate::new("cherry", false),
                Candidate::new("date", false),
                Candidate::new("eggplant", false),
            ])
            .ballots([
                Ballot::new(1, [vec![0, 1], vec![2, 3, 4]]),
                Ballot::new(2, [vec![1, 2], vec![3, 4, 0]]),
                Ballot::new(3, [vec![2, 3], vec![4, 0, 1]]),
                Ballot::new(4, [vec![3, 4], vec![0, 1, 2]]),
                Ballot::new(5, [vec![4, 0], vec![1, 2, 3]]),
            ])
            .build();

        let mut buf = Vec::new();
        let result = stv_droop::<Integer64, FixedDecimal9>(
            &mut buf,
            &election,
            "package name",
            6,
            parallel,
            None,
            false,
            false,
            true,
        )
        .unwrap();

        assert_eq!(
            result,
            ElectionResult {
                elected: vec![4, 3, 0],
                not_elected: vec![1, 2],
                transcript: vec![
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(1500000000),
                            FixedDecimal9::new(2500000000),
                            FixedDecimal9::new(3500000000),
                            FixedDecimal9::new(4500000000)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    StartRound(1),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(1500000000),
                            FixedDecimal9::new(2500000000),
                            FixedDecimal9::new(3500000000),
                            FixedDecimal9::new(4500000000)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    Quota(FixedDecimal9::new(3750000001)),
                    Elected(4, ElectedReason::ExceedsQuota),
                    Surplus(FixedDecimal9::new(749999999)),
                    StartRound(2),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3000000000),
                            FixedDecimal9::new(1500000000),
                            FixedDecimal9::new(2500000000),
                            FixedDecimal9::new(3500000000),
                            FixedDecimal9::new(4500000000)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    Quota(FixedDecimal9::new(3750000001)),
                    Surplus(FixedDecimal9::new(749999999)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(833333334)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3416666665),
                            FixedDecimal9::new(1500000000),
                            FixedDecimal9::new(2500000000),
                            FixedDecimal9::new(3833333332),
                            FixedDecimal9::new(3750000003)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    Quota(FixedDecimal9::new(3750000001)),
                    Elected(3, ElectedReason::ExceedsQuota),
                    Surplus(FixedDecimal9::new(83333333)),
                    StartRound(3),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3416666665),
                            FixedDecimal9::new(1500000000),
                            FixedDecimal9::new(2500000000),
                            FixedDecimal9::new(3833333332),
                            FixedDecimal9::new(3750000003)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    Quota(FixedDecimal9::new(3750000001)),
                    Surplus(FixedDecimal9::new(83333333)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(978260871),
                        FixedDecimal9::new(833333334)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3421497581),
                            FixedDecimal9::new(1504830916),
                            FixedDecimal9::new(2537439608),
                            FixedDecimal9::new(3750000001),
                            FixedDecimal9::new(3786231883)
                        ],
                        exhausted: FixedDecimal9::new(11)
                    },
                    Quota(FixedDecimal9::new(3749999998)),
                    Surplus(FixedDecimal9::new(36231888)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(978260871),
                        FixedDecimal9::new(825358853)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3441664925),
                            FixedDecimal9::new(1505062060),
                            FixedDecimal9::new(2537670752),
                            FixedDecimal9::new(3765602245),
                            FixedDecimal9::new(3749999998)
                        ],
                        exhausted: FixedDecimal9::new(20)
                    },
                    Quota(FixedDecimal9::new(3749999996)),
                    Surplus(FixedDecimal9::new(15602251)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(974207584),
                        FixedDecimal9::new(825358853)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3442608753),
                            FixedDecimal9::new(1506005888),
                            FixedDecimal9::new(2544694512),
                            FixedDecimal9::new(3750000000),
                            FixedDecimal9::new(3756690830)
                        ],
                        exhausted: FixedDecimal9::new(17)
                    },
                    Quota(FixedDecimal9::new(3749999996)),
                    Surplus(FixedDecimal9::new(6690838)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(974207584),
                        FixedDecimal9::new(823888853)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3446334305),
                            FixedDecimal9::new(1506056440),
                            FixedDecimal9::new(2544745064),
                            FixedDecimal9::new(3752864172),
                            FixedDecimal9::new(3750000002)
                        ],
                        exhausted: FixedDecimal9::new(17)
                    },
                    Quota(FixedDecimal9::new(3749999996)),
                    Surplus(FixedDecimal9::new(2864182)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973464072),
                        FixedDecimal9::new(823888852)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3446508898),
                            FixedDecimal9::new(1506231028),
                            FixedDecimal9::new(2546034920),
                            FixedDecimal9::new(3750000000),
                            FixedDecimal9::new(3751225142)
                        ],
                        exhausted: FixedDecimal9::new(12)
                    },
                    Quota(FixedDecimal9::new(3749999998)),
                    Surplus(FixedDecimal9::new(1225146)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973464072),
                        FixedDecimal9::new(823619772)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3447191118),
                            FixedDecimal9::new(1506240548),
                            FixedDecimal9::new(2546044440),
                            FixedDecimal9::new(3750523880),
                            FixedDecimal9::new(3750000002)
                        ],
                        exhausted: FixedDecimal9::new(12)
                    },
                    Quota(FixedDecimal9::new(3749999998)),
                    Surplus(FixedDecimal9::new(523886)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973328097),
                        FixedDecimal9::new(823619772)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3447223094),
                            FixedDecimal9::new(1506272524),
                            FixedDecimal9::new(2546280377),
                            FixedDecimal9::new(3750000000),
                            FixedDecimal9::new(3750223986)
                        ],
                        exhausted: FixedDecimal9::new(19)
                    },
                    Quota(FixedDecimal9::new(3749999996)),
                    Surplus(FixedDecimal9::new(223994)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973328096),
                        FixedDecimal9::new(823570580)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3447347826),
                            FixedDecimal9::new(1506274276),
                            FixedDecimal9::new(2546282132),
                            FixedDecimal9::new(3750095756),
                            FixedDecimal9::new(3749999998)
                        ],
                        exhausted: FixedDecimal9::new(12)
                    },
                    Quota(FixedDecimal9::new(3749999998)),
                    Surplus(FixedDecimal9::new(95758)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973303243),
                        FixedDecimal9::new(823570581)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3447353665),
                            FixedDecimal9::new(1506280120),
                            FixedDecimal9::new(2546325254),
                            FixedDecimal9::new(3749999999),
                            FixedDecimal9::new(3750040938)
                        ],
                        exhausted: FixedDecimal9::new(24)
                    },
                    Quota(FixedDecimal9::new(3749999995)),
                    Surplus(FixedDecimal9::new(40947)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973303243),
                        FixedDecimal9::new(823561590)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3447376465),
                            FixedDecimal9::new(1506280440),
                            FixedDecimal9::new(2546325574),
                            FixedDecimal9::new(3750017499),
                            FixedDecimal9::new(3749999999)
                        ],
                        exhausted: FixedDecimal9::new(23)
                    },
                    Quota(FixedDecimal9::new(3749999995)),
                    Surplus(FixedDecimal9::new(17508)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973298701),
                        FixedDecimal9::new(823561590)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3447377533),
                            FixedDecimal9::new(1506281508),
                            FixedDecimal9::new(2546333455),
                            FixedDecimal9::new(3750000002),
                            FixedDecimal9::new(3750007483)
                        ],
                        exhausted: FixedDecimal9::new(19)
                    },
                    Quota(FixedDecimal9::new(3749999996)),
                    Surplus(FixedDecimal9::new(7493)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973298700),
                        FixedDecimal9::new(823559946)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3447381703),
                            FixedDecimal9::new(1506281568),
                            FixedDecimal9::new(2546333518),
                            FixedDecimal9::new(3750003198),
                            FixedDecimal9::new(3749999997)
                        ],
                        exhausted: FixedDecimal9::new(16)
                    },
                    Quota(FixedDecimal9::new(3749999997)),
                    Surplus(FixedDecimal9::new(3201)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973297870),
                        FixedDecimal9::new(823559947)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3447381894),
                            FixedDecimal9::new(1506281764),
                            FixedDecimal9::new(2546334959),
                            FixedDecimal9::new(3749999997),
                            FixedDecimal9::new(3750001365)
                        ],
                        exhausted: FixedDecimal9::new(21)
                    },
                    Quota(FixedDecimal9::new(3749999995)),
                    Surplus(FixedDecimal9::new(1372)),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973297870),
                        FixedDecimal9::new(823559647)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3447382656),
                            FixedDecimal9::new(1506281776),
                            FixedDecimal9::new(2546334971),
                            FixedDecimal9::new(3750000581),
                            FixedDecimal9::new(3749999999)
                        ],
                        exhausted: FixedDecimal9::new(17)
                    },
                    Quota(FixedDecimal9::new(3749999996)),
                    Surplus(FixedDecimal9::new(588)),
                    NotElected(1, NotElectedReason::LowSurplus),
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(0),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(973297870),
                        FixedDecimal9::new(823559647)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3950523544),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3549475859),
                            FixedDecimal9::new(3750000581),
                            FixedDecimal9::new(3749999999)
                        ],
                        exhausted: FixedDecimal9::new(17)
                    },
                    StartRound(4),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(3950523544),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(3549475859),
                            FixedDecimal9::new(3750000581),
                            FixedDecimal9::new(3749999999)
                        ],
                        exhausted: FixedDecimal9::new(17)
                    },
                    Quota(FixedDecimal9::new(3749999996)),
                    Elected(0, ElectedReason::ExceedsQuota),
                    Surplus(FixedDecimal9::new(200524136)),
                    NotElected(2, NotElectedReason::Remaining),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(4744588121),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(5916055638),
                            FixedDecimal9::new(4339356223)
                        ],
                        exhausted: FixedDecimal9::new(18)
                    }
                ],
            },
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	package name
	Rule: Meek Parametric (omega = 1/10^6)
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 3
	Ballots: 15
	Quota: 3.750000001
	Omega: 0.000001000

	Add eligible: Apple
	Add eligible: Banana
	Add eligible: Cherry
	Add eligible: Date
	Add eligible: Eggplant
Action: Begin Count
	Hopeful:  Apple (3.000000000)
	Hopeful:  Banana (1.500000000)
	Hopeful:  Cherry (2.500000000)
	Hopeful:  Date (3.500000000)
	Hopeful:  Eggplant (4.500000000)
	Quota: 3.750000001
	Votes: 15.000000000
	Residual: 0.000000000
	Total: 15.000000000
	Surplus: 0.000000000
Round 1:
Action: Elect: Eggplant
	Elected:  Eggplant (4.500000000)
	Hopeful:  Apple (3.000000000)
	Hopeful:  Banana (1.500000000)
	Hopeful:  Cherry (2.500000000)
	Hopeful:  Date (3.500000000)
	Quota: 3.750000001
	Votes: 15.000000000
	Residual: 0.000000000
	Total: 15.000000000
	Surplus: 0.000000000
Action: Iterate (elected)
	Quota: 3.750000001
	Votes: 15.000000000
	Residual: 0.000000000
	Total: 15.000000000
	Surplus: 0.749999999
Round 2:
Action: Elect: Date
	Elected:  Date (3.833333332)
	Elected:  Eggplant (3.750000003)
	Hopeful:  Apple (3.416666665)
	Hopeful:  Banana (1.500000000)
	Hopeful:  Cherry (2.500000000)
	Quota: 3.750000001
	Votes: 15.000000000
	Residual: 0.000000000
	Total: 15.000000000
	Surplus: 0.749999999
Action: Iterate (elected)
	Quota: 3.750000001
	Votes: 15.000000000
	Residual: 0.000000000
	Total: 15.000000000
	Surplus: 0.083333333
Round 3:
Action: Iterate (omega)
	Quota: 3.749999996
	Votes: 14.999999983
	Residual: 0.000000017
	Total: 15.000000000
	Surplus: 0.000000588
Action: Defeat (surplus 0.000000588 < omega): Banana
	Elected:  Date (3.750000581)
	Elected:  Eggplant (3.749999999)
	Hopeful:  Apple (3.447382656)
	Hopeful:  Cherry (2.546334971)
	Defeated: Banana (1.506281776)
	Quota: 3.749999996
	Votes: 14.999999983
	Residual: 0.000000017
	Total: 15.000000000
	Surplus: 0.000000588
Round 4:
Action: Elect: Apple
	Elected:  Apple (3.950523544)
	Elected:  Date (3.750000581)
	Elected:  Eggplant (3.749999999)
	Hopeful:  Cherry (3.549475859)
	Defeated: Banana (0.000000000)
	Quota: 3.749999996
	Votes: 14.999999983
	Residual: 0.000000017
	Total: 15.000000000
	Surplus: 0.000000588
Action: Iterate (elected)
	Quota: 3.749999996
	Votes: 14.999999983
	Residual: 0.000000017
	Total: 15.000000000
	Surplus: 0.200524136
Action: Defeat remaining: Cherry
	Elected:  Apple (3.950523544)
	Elected:  Date (3.750000581)
	Elected:  Eggplant (3.749999999)
	Defeated: Cherry (3.549475859)
	Defeated: Banana (0.000000000)
	Quota: 3.749999996
	Votes: 14.999999983
	Residual: 0.000000017
	Total: 15.000000000
	Surplus: 0.200524136
Action: Count Complete
	Elected:  Apple (4.744588121)
	Elected:  Date (5.916055638)
	Elected:  Eggplant (4.339356223)
	Defeated: Banana, Cherry (0.000000000)
	Quota: 3.749999996
	Votes: 14.999999982
	Residual: 0.000000018
	Total: 15.000000000
	Surplus: 0.200524136

"
        );
    }

    #[test]
    fn test_stv_droop_below_quota() {
        let election = Election::builder()
            .title("Vegetable contest")
            .num_seats(2)
            .candidates([
                Candidate::new("apple", false),
                Candidate::new("banana", false),
                Candidate::new("cherry", false),
                Candidate::new("date", false),
            ])
            .ballots([
                Ballot::new(3, [vec![0, 1], vec![2]]),
                Ballot::new(2, [vec![0, 3], vec![1]]),
            ])
            .build();

        let logger = ThreadLocalLogger::start();
        let mut buf = Vec::new();
        let result = stv_droop::<Integer64, FixedDecimal9>(
            &mut buf,
            &election,
            "package name",
            6,
            Parallel::No,
            None,
            false,
            false,
            false,
        )
        .unwrap();

        assert_eq!(
            result,
            ElectionResult {
                elected: vec![0, 1],
                not_elected: vec![2, 3],
                transcript: vec![
                    KeepFactors(vec![
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(2500000000),
                            FixedDecimal9::new(1500000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(1000000000)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    StartRound(1),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(2500000000),
                            FixedDecimal9::new(1500000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(1000000000)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    Quota(FixedDecimal9::new(1666666667)),
                    Elected(0, ElectedReason::ExceedsQuota),
                    Surplus(FixedDecimal9::new(833333333)),
                    StartRound(2),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(2500000000),
                            FixedDecimal9::new(1500000000),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(1000000000)
                        ],
                        exhausted: FixedDecimal9::new(0)
                    },
                    Quota(FixedDecimal9::new(1666666667)),
                    Surplus(FixedDecimal9::new(833333333)),
                    KeepFactors(vec![
                        FixedDecimal9::new(666666667),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000),
                        FixedDecimal9::new(1000000000)
                    ]),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(1666666665),
                            FixedDecimal9::new(1833333332),
                            FixedDecimal9::new(499999998),
                            FixedDecimal9::new(1000000000)
                        ],
                        exhausted: FixedDecimal9::new(5)
                    },
                    Quota(FixedDecimal9::new(1666666666)),
                    Elected(1, ElectedReason::ExceedsQuota),
                    Surplus(FixedDecimal9::new(166666665)),
                    NotElected(2, NotElectedReason::Remaining),
                    NotElected(3, NotElectedReason::Remaining),
                    Count {
                        tallies: vec![
                            FixedDecimal9::new(2333333333),
                            FixedDecimal9::new(2166666666),
                            FixedDecimal9::new(0),
                            FixedDecimal9::new(0)
                        ],
                        exhausted: FixedDecimal9::new(500000001)
                    }
                ],
            },
        );
        logger.check_logs_at_target_level(
            "stv_rs::meek",
            Warn,
            r"Count for elected candidate #0 (apple) is lower than the quota: 1.666666665 < 1.666666666 ~ 1.666666665 < 1.666666666
",
        );

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	package name
	Rule: Meek Parametric (omega = 1/10^6)
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 2
	Ballots: 5
	Quota: 1.666666667
	Omega: 0.000001000

	Add eligible: Apple
	Add eligible: Banana
	Add eligible: Cherry
	Add eligible: Date
Action: Begin Count
	Hopeful:  Apple (2.500000000)
	Hopeful:  Banana (1.500000000)
	Hopeful:  Cherry (0.000000000)
	Hopeful:  Date (1.000000000)
	Quota: 1.666666667
	Votes: 5.000000000
	Residual: 0.000000000
	Total: 5.000000000
	Surplus: 0.000000000
Round 1:
Action: Elect: Apple
	Elected:  Apple (2.500000000)
	Hopeful:  Banana (1.500000000)
	Hopeful:  Cherry (0.000000000)
	Hopeful:  Date (1.000000000)
	Quota: 1.666666667
	Votes: 5.000000000
	Residual: 0.000000000
	Total: 5.000000000
	Surplus: 0.000000000
Action: Iterate (elected)
	Quota: 1.666666667
	Votes: 5.000000000
	Residual: 0.000000000
	Total: 5.000000000
	Surplus: 0.833333333
Round 2:
Action: Elect: Banana
	Elected:  Apple (1.666666665)
	Elected:  Banana (1.833333332)
	Hopeful:  Cherry (0.499999998)
	Hopeful:  Date (1.000000000)
	Quota: 1.666666666
	Votes: 4.999999995
	Residual: 0.000000005
	Total: 5.000000000
	Surplus: 0.833333333
Action: Iterate (elected)
	Quota: 1.666666666
	Votes: 4.999999995
	Residual: 0.000000005
	Total: 5.000000000
	Surplus: 0.166666665
Action: Defeat remaining: Cherry
	Elected:  Apple (1.666666665)
	Elected:  Banana (1.833333332)
	Hopeful:  Date (1.000000000)
	Defeated: Cherry (0.499999998)
	Quota: 1.666666666
	Votes: 4.999999995
	Residual: 0.000000005
	Total: 5.000000000
	Surplus: 0.166666665
Action: Defeat remaining: Date
	Elected:  Apple (1.666666665)
	Elected:  Banana (1.833333332)
	Defeated: Date (1.000000000)
	Defeated: Cherry (0.000000000)
	Quota: 1.666666666
	Votes: 4.499999997
	Residual: 0.500000003
	Total: 5.000000000
	Surplus: 0.166666665
Action: Count Complete
	Elected:  Apple (2.333333333)
	Elected:  Banana (2.166666666)
	Defeated: Cherry, Date (0.000000000)
	Quota: 1.666666666
	Votes: 4.499999999
	Residual: 0.500000001
	Total: 5.000000000
	Surplus: 0.166666665

"
        );
    }

    #[cfg(feature = "checked_i64")]
    fn make_panicking_election() -> Election {
        Election::builder()
            .title("Vegetable contest")
            .num_seats(1)
            .candidates([Candidate::new("apple", false)])
            .ballots([
                Ballot::new(9223372036854775807, [vec![0]]),
                Ballot::new(9223372036854775807, [vec![0]]),
                Ballot::new(9223372036854775807, [vec![0]]),
                Ballot::new(9223372036854775807, [vec![0]]),
                Ballot::new(9223372036854775807, [vec![0]]),
            ])
            .build()
    }

    #[cfg(feature = "checked_i64")]
    #[test]
    #[cfg_attr(
        not(overflow_checks),
        should_panic(expected = "called `Option::unwrap()` on a `None` value")
    )]
    #[cfg_attr(
        overflow_checks,
        should_panic(expected = "attempt to add with overflow")
    )]
    fn test_stv_droop_panic() {
        let _ = stv_droop::<Integer64, FixedDecimal9>(
            &mut Vec::new(),
            &make_panicking_election(),
            "package name",
            6,
            Parallel::Custom,
            /* num_threads = */ Some(NonZeroUsize::new(2).unwrap()),
            false,
            false,
            false,
        );
    }

    #[cfg(feature = "checked_i64")]
    #[test]
    fn test_stv_droop_panic_logs() {
        let logger = ThreadLocalLogger::start();
        let result = std::panic::catch_unwind(|| {
            stv_droop::<Integer64, FixedDecimal9>(
                &mut Vec::new(),
                &make_panicking_election(),
                "package name",
                6,
                Parallel::Custom,
                /* num_threads = */ Some(NonZeroUsize::new(2).unwrap()),
                false,
                false,
                false,
            )
        });

        // Unfortunately, Rust's test harness and/or the `catch_unwind()`
        // function seem to interfere with panic propagation, so we cannot test
        // all the details of the double-panic (worker thread + main thread)
        // here in a unit test.
        assert!(result.is_err());
        assert_eq!(
            result.as_ref().unwrap_err().type_id(),
            std::any::TypeId::of::<&str>()
        );

        #[cfg(not(overflow_checks))]
        assert_eq!(
            *result.unwrap_err().downcast::<&str>().unwrap(),
            "called `Option::unwrap()` on a `None` value"
        );
        #[cfg(overflow_checks)]
        assert_eq!(
            *result.unwrap_err().downcast::<&str>().unwrap(),
            "attempt to add with overflow"
        );

        #[cfg(not(overflow_checks))]
        logger.check_logs([
            (
                Info,
                "stv_rs::meek",
                "Parallel vote counting is enabled using custom threads",
            ),
            (Info, "stv_rs::meek", "Equalized vote counting is disabled"),
            (Info, "stv_rs::meek", "Spawning 2 threads"),
            (
                Debug,
                "stv_rs::parallelism::thread_pool",
                "[main thread] Spawned threads",
            ),
            (
                Debug,
                "stv_rs::parallelism::thread_pool",
                "[main thread] Notifying threads to finish...",
            ),
            (
                Debug,
                "stv_rs::parallelism::thread_pool",
                "[main thread] Joining threads in the pool...",
            ),
            (
                Debug,
                "stv_rs::parallelism::thread_pool",
                "[main thread] Thread 0 joined with result: Ok(())",
            ),
            (
                Debug,
                "stv_rs::parallelism::thread_pool",
                "[main thread] Thread 1 joined with result: Ok(())",
            ),
            (
                Debug,
                "stv_rs::parallelism::thread_pool",
                "[main thread] Joined threads.",
            ),
            #[cfg(feature = "log_parallelism")]
            (
                Info,
                "stv_rs::parallelism::range",
                "Work-stealing statistics:",
            ),
            #[cfg(feature = "log_parallelism")]
            (Info, "stv_rs::parallelism::range", "- increments: 0"),
            #[cfg(feature = "log_parallelism")]
            (Info, "stv_rs::parallelism::range", "- failed_increments: 0"),
            #[cfg(feature = "log_parallelism")]
            (Info, "stv_rs::parallelism::range", "- other_loads: 0"),
            #[cfg(feature = "log_parallelism")]
            (Info, "stv_rs::parallelism::range", "- thefts: 0"),
            #[cfg(feature = "log_parallelism")]
            (Info, "stv_rs::parallelism::range", "- failed_thefts: 0"),
            #[cfg(feature = "log_parallelism")]
            (
                Info,
                "stv_rs::parallelism::range",
                "- increments + thefts: 0",
            ),
        ]);
        #[cfg(overflow_checks)]
        logger.check_logs([]);
    }

    #[test]
    fn test_start_election() {
        let election = Election::builder()
            .title("Vegetable contest")
            .num_seats(3)
            .candidates([
                Candidate::new("apple", false),
                Candidate::new("banana", true),
                Candidate::new("cherry", false),
                Candidate::new("date", false),
                Candidate::new("eggplant", false),
                Candidate::new("fig", true),
                Candidate::new("grape", false),
                Candidate::new("hazelnut", false),
                Candidate::new("jalapeno", false),
            ])
            .ballots([
                Ballot::new(1, [vec![0]]),
                Ballot::new(2, [vec![2]]),
                Ballot::new(3, [vec![3]]),
                Ballot::new(4, [vec![4]]),
                Ballot::new(5, [vec![6]]),
                Ballot::new(6, [vec![7]]),
                Ballot::new(7, [vec![8]]),
            ])
            .build();
        let omega_exponent = 6;
        let state = State::new(&election, omega_exponent, Parallel::No, false, None, None);

        let mut buf = Vec::new();
        let count = state
            .start_election(&mut buf, "STV-rs", omega_exponent)
            .unwrap();

        assert_eq!(
            count,
            VoteCount::new(
                [
                    FixedDecimal9::from_usize(1),
                    FixedDecimal9::zero(),
                    FixedDecimal9::from_usize(2),
                    FixedDecimal9::from_usize(3),
                    FixedDecimal9::from_usize(4),
                    FixedDecimal9::zero(),
                    FixedDecimal9::from_usize(5),
                    FixedDecimal9::from_usize(6),
                    FixedDecimal9::from_usize(7),
                ],
                FixedDecimal9::zero(),
            )
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	STV-rs
	Rule: Meek Parametric (omega = 1/10^6)
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 3
	Ballots: 28
	Quota: 7.000000001
	Omega: 0.000001000

	Add eligible: Apple
	Add withdrawn: Banana
	Add eligible: Cherry
	Add eligible: Date
	Add eligible: Eggplant
	Add withdrawn: Fig
	Add eligible: Grape
	Add eligible: Hazelnut
	Add eligible: Jalapeno
Action: Begin Count
	Hopeful:  Apple (1.000000000)
	Hopeful:  Cherry (2.000000000)
	Hopeful:  Date (3.000000000)
	Hopeful:  Eggplant (4.000000000)
	Hopeful:  Grape (5.000000000)
	Hopeful:  Hazelnut (6.000000000)
	Hopeful:  Jalapeno (7.000000000)
	Quota: 7.000000001
	Votes: 28.000000000
	Residual: 0.000000000
	Total: 28.000000000
	Surplus: 0.000000000
"
        );
    }

    #[test]
    fn test_election_completed() {
        fn make_election(num_seats: usize) -> Election {
            Election::builder()
                .title("Vegetable contest")
                .num_seats(num_seats)
                .candidates([
                    Candidate::new("apple", false),
                    Candidate::new("banana", true),
                    Candidate::new("cherry", false),
                    Candidate::new("date", false),
                    Candidate::new("eggplant", false),
                ])
                .build()
        }

        fn make_state(
            election: &Election,
            statuses: [Status; 5],
        ) -> State<'_, '_, Integer64, FixedDecimal9> {
            State::builder()
                .election(election)
                .statuses(statuses)
                .keep_factors([FixedDecimal9::zero(); 5])
                .threshold(FixedDecimal9::zero())
                .surplus(FixedDecimal9::zero())
                .omega(FixedDecimal9::zero())
                .parallel(Parallel::No)
                .force_positive_surplus(false)
                .pascal(None)
                .build()
        }

        // All remaining candidates should be defeated.
        let logger = ThreadLocalLogger::start();
        let election = make_election(1);
        let state = make_state(
            &election,
            [
                Status::Candidate,
                Status::Withdrawn,
                Status::Elected,
                Status::NotElected,
                Status::Candidate,
            ],
        );

        let mut buf = Vec::new();
        let completed = state.election_completed(&mut buf, 42).unwrap();

        assert!(completed);
        assert!(buf.is_empty());
        logger.check_target_logs(
            "stv_rs::meek",
            [
                (
                    Debug,
                    "Checking if count is complete: candidates=2, to_elect=0",
                ),
                (Info, "Count is now complete!"),
            ],
        );

        // All remaining candidates should be elected.
        let logger = ThreadLocalLogger::start();
        let election = make_election(3);
        let state = make_state(
            &election,
            [
                Status::Candidate,
                Status::Withdrawn,
                Status::Elected,
                Status::NotElected,
                Status::Candidate,
            ],
        );

        let mut buf = Vec::new();
        let completed = state.election_completed(&mut buf, 42).unwrap();

        assert!(completed);
        assert!(buf.is_empty());
        logger.check_target_logs(
            "stv_rs::meek",
            [
                (
                    Debug,
                    "Checking if count is complete: candidates=2, to_elect=2",
                ),
                (Info, "Count is now complete!"),
            ],
        );

        // The election is not completed yet.
        let logger = ThreadLocalLogger::start();
        let election = make_election(2);
        let state = make_state(
            &election,
            [
                Status::Candidate,
                Status::Withdrawn,
                Status::Elected,
                Status::NotElected,
                Status::Candidate,
            ],
        );

        let mut buf = Vec::new();
        let completed = state.election_completed(&mut buf, 42).unwrap();

        assert!(!completed);
        assert_eq!(std::str::from_utf8(&buf).unwrap(), "Round 42:\n");
        logger.check_target_level_logs(
            "stv_rs::meek",
            Debug,
            r"Checking if count is complete: candidates=2, to_elect=1
Weights:
    [0] apple (Candidate) ~ 0
    [1] banana (Withdrawn) ~ 0
    [2] cherry (Elected) ~ 0
    [3] date (NotElected) ~ 0
    [4] eggplant (Candidate) ~ 0
",
        );
    }

    #[test]
    fn test_election_round() {
        fn make_election(ballots: impl Into<Vec<Ballot>>) -> Election {
            Election::builder()
                .title("Vegetable contest")
                .num_seats(2)
                .candidates([
                    Candidate::new("apple", false),
                    Candidate::new("banana", false),
                    Candidate::new("cherry", false),
                    Candidate::new("date", false),
                ])
                .ballots(ballots)
                .build()
        }

        fn make_state(
            election: &Election,
            statuses: impl Into<Vec<Status>>,
        ) -> State<'_, '_, Integer64, FixedDecimal9> {
            State::builder()
                .election(election)
                .statuses(statuses)
                .keep_factors([FixedDecimal9::one(); 4])
                .threshold(FixedDecimal9::zero())
                .surplus(FixedDecimal9::zero())
                .omega(FixedDecimal9::ratio(1, 1_000_000))
                .parallel(Parallel::No)
                .force_positive_surplus(false)
                .pascal(None)
                .build()
        }

        fn make_count() -> VoteCount<Integer64, FixedDecimal9> {
            VoteCount::new([FixedDecimal9::zero(); 4], FixedDecimal9::zero())
        }

        // No ballot.
        let election = make_election([]);
        let mut state = make_state(&election, [Status::Candidate; 4]);
        let mut count = make_count();

        let mut buf = Vec::new();
        state.election_round(&mut buf, &mut count).unwrap();
        assert_eq!(
            (
                state.threshold,
                state.surplus,
                state.statuses,
                state.keep_factors
            ),
            (
                FixedDecimal9::epsilon(),
                FixedDecimal9::zero(),
                vec![
                    Status::NotElected,
                    Status::Candidate,
                    Status::Candidate,
                    Status::Candidate,
                ],
                vec![
                    FixedDecimal9::zero(),
                    FixedDecimal9::one(),
                    FixedDecimal9::one(),
                    FixedDecimal9::one(),
                ]
            )
        );
        assert_eq!(
            count,
            VoteCount::new([FixedDecimal9::zero(); 4], FixedDecimal9::zero())
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"Action: Iterate (omega)
	Quota: 0.000000001
	Votes: 0.000000000
	Residual: 0.000000000
	Total: 0.000000000
	Surplus: 0.000000000
Action: Break tie (defeat): [Apple, Banana, Cherry, Date] -> Apple
	Quota: 0.000000001
	Votes: 0.000000000
	Residual: 0.000000000
	Total: 0.000000000
	Surplus: 0.000000000
Action: Defeat (surplus 0.000000000 < omega): Apple
	Hopeful:  Banana (0.000000000)
	Hopeful:  Cherry (0.000000000)
	Hopeful:  Date (0.000000000)
	Defeated: Apple (0.000000000)
	Quota: 0.000000001
	Votes: 0.000000000
	Residual: 0.000000000
	Total: 0.000000000
	Surplus: 0.000000000
"
        );

        // One candidate is elected.
        let election = make_election([Ballot::new(1, [vec![1]])]);
        let mut state = make_state(&election, [Status::Candidate; 4]);
        let mut count = make_count();

        let mut buf = Vec::new();
        state.election_round(&mut buf, &mut count).unwrap();
        assert_eq!(
            (
                state.threshold,
                state.surplus,
                state.statuses,
                state.keep_factors
            ),
            (
                FixedDecimal9::ratio(1, 3) + FixedDecimal9::epsilon(),
                FixedDecimal9::ratio(2, 3),
                vec![
                    Status::Candidate,
                    Status::Elected,
                    Status::Candidate,
                    Status::Candidate,
                ],
                vec![FixedDecimal9::one(); 4]
            )
        );
        assert_eq!(
            count,
            VoteCount::new(
                [
                    FixedDecimal9::zero(),
                    FixedDecimal9::one(),
                    FixedDecimal9::zero(),
                    FixedDecimal9::zero(),
                ],
                FixedDecimal9::zero(),
            )
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"Action: Elect: Banana
	Elected:  Banana (1.000000000)
	Hopeful:  Apple (0.000000000)
	Hopeful:  Cherry (0.000000000)
	Hopeful:  Date (0.000000000)
	Quota: 0.333333334
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000
Action: Iterate (elected)
	Quota: 0.333333334
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.666666666
"
        );

        // Omega threshold.
        let election = make_election([
            Ballot::new(100, [vec![1]]),
            Ballot::new(1, [vec![0]]),
            Ballot::new(2, [vec![2]]),
            Ballot::new(3, [vec![3]]),
        ]);
        let mut state = make_state(
            &election,
            [
                Status::Candidate,
                Status::Elected,
                Status::Candidate,
                Status::Candidate,
            ],
        );
        let mut count = make_count();

        let mut buf = Vec::new();
        state.election_round(&mut buf, &mut count).unwrap();
        assert_eq!(
            (
                state.threshold,
                state.surplus,
                state.statuses,
                state.keep_factors
            ),
            (
                FixedDecimal9::new(3_000_000_301),
                FixedDecimal9::new(599),
                vec![
                    Status::NotElected,
                    Status::Elected,
                    Status::Candidate,
                    Status::Candidate,
                ],
                vec![
                    FixedDecimal9::zero(),
                    FixedDecimal9::new(30_000_009),
                    FixedDecimal9::one(),
                    FixedDecimal9::one(),
                ]
            )
        );
        assert_eq!(
            count,
            VoteCount::new(
                [
                    FixedDecimal9::zero(),
                    FixedDecimal9::new(3_000_000_900),
                    FixedDecimal9::from_usize(2),
                    FixedDecimal9::from_usize(3),
                ],
                FixedDecimal9::new(97_999_999_100),
            )
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"Action: Iterate (omega)
	Quota: 3.000000301
	Votes: 9.000000900
	Residual: 96.999999100
	Total: 106.000000000
	Surplus: 0.000000599
Action: Defeat (surplus 0.000000599 < omega): Apple
	Elected:  Banana (3.000000900)
	Hopeful:  Cherry (2.000000000)
	Hopeful:  Date (3.000000000)
	Defeated: Apple (1.000000000)
	Quota: 3.000000301
	Votes: 9.000000900
	Residual: 96.999999100
	Total: 106.000000000
	Surplus: 0.000000599
"
        );

        // Stable iteration.
        let election = make_election([
            Ballot::new(100_000, [vec![1]]),
            Ballot::new(1, [vec![0]]),
            Ballot::new(2, [vec![2]]),
            Ballot::new(3, [vec![3]]),
        ]);
        let mut state = make_state(
            &election,
            [
                Status::Candidate,
                Status::Elected,
                Status::Candidate,
                Status::Candidate,
            ],
        );
        let mut count = make_count();

        let mut buf = Vec::new();
        state.election_round(&mut buf, &mut count).unwrap();
        assert_eq!(
            (
                state.threshold,
                state.surplus,
                state.statuses,
                state.keep_factors
            ),
            (
                FixedDecimal9::new(3_000_033_334),
                FixedDecimal9::new(66_666),
                vec![
                    Status::NotElected,
                    Status::Elected,
                    Status::Candidate,
                    Status::Candidate,
                ],
                vec![
                    FixedDecimal9::zero(),
                    FixedDecimal9::new(30_001),
                    FixedDecimal9::one(),
                    FixedDecimal9::one(),
                ]
            )
        );
        assert_eq!(
            count,
            VoteCount::new(
                [
                    FixedDecimal9::zero(),
                    FixedDecimal9::new(3_000_100_000),
                    FixedDecimal9::from_usize(2),
                    FixedDecimal9::from_usize(3),
                ],
                FixedDecimal9::new(99_997_999_900_000),
            )
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"	Stable state detected (0.000066666)
Action: Iterate (stable)
	Quota: 3.000033334
	Votes: 9.000100000
	Residual: 99996.999900000
	Total: 100006.000000000
	Surplus: 0.000066666
Action: Defeat (stable surplus 0.000066666): Apple
	Elected:  Banana (3.000100000)
	Hopeful:  Cherry (2.000000000)
	Hopeful:  Date (3.000000000)
	Defeated: Apple (1.000000000)
	Quota: 3.000033334
	Votes: 9.000100000
	Residual: 99996.999900000
	Total: 100006.000000000
	Surplus: 0.000066666
"
        );
    }

    #[test]
    fn test_iterate_droop() {
        fn make_election(ballots: impl Into<Vec<Ballot>>) -> Election {
            Election::builder()
                .title("Vegetable contest")
                .num_seats(2)
                .candidates([
                    Candidate::new("apple", false),
                    Candidate::new("banana", false),
                    Candidate::new("cherry", false),
                    Candidate::new("date", false),
                ])
                .ballots(ballots)
                .build()
        }

        fn make_state(
            election: &Election,
            statuses: impl Into<Vec<Status>>,
        ) -> State<'_, '_, Integer64, FixedDecimal9> {
            State::builder()
                .election(election)
                .statuses(statuses)
                .keep_factors([FixedDecimal9::one(); 4])
                .threshold(FixedDecimal9::zero())
                .surplus(FixedDecimal9::zero())
                .omega(FixedDecimal9::ratio(1, 1_000_000))
                .parallel(Parallel::No)
                .force_positive_surplus(false)
                .pascal(None)
                .build()
        }

        fn make_count() -> VoteCount<Integer64, FixedDecimal9> {
            VoteCount::new([FixedDecimal9::zero(); 4], FixedDecimal9::zero())
        }

        // No ballot.
        let election = make_election([]);
        let mut state = make_state(&election, [Status::Candidate; 4]);
        let mut count = make_count();

        let mut buf = Vec::new();
        let iteration = state.iterate_droop(&mut buf, &mut count).unwrap();
        assert_eq!(iteration, DroopIteration::Omega);
        // TODO: Use expectations rather than assertions.
        assert_eq!(
            (
                state.threshold,
                state.surplus,
                state.statuses,
                state.keep_factors
            ),
            (
                FixedDecimal9::epsilon(),
                FixedDecimal9::zero(),
                vec![Status::Candidate; 4],
                vec![FixedDecimal9::one(); 4]
            )
        );
        assert_eq!(
            count,
            VoteCount::new([FixedDecimal9::zero(); 4], FixedDecimal9::zero())
        );
        assert!(buf.is_empty());

        // One candidate is elected.
        let election = make_election([Ballot::new(1, [vec![1]])]);
        let mut state = make_state(&election, [Status::Candidate; 4]);
        let mut count = make_count();

        let mut buf = Vec::new();
        let iteration = state.iterate_droop(&mut buf, &mut count).unwrap();
        assert_eq!(iteration, DroopIteration::Elected);
        assert_eq!(
            (
                state.threshold,
                state.surplus,
                state.statuses,
                state.keep_factors
            ),
            (
                FixedDecimal9::ratio(1, 3) + FixedDecimal9::epsilon(),
                FixedDecimal9::ratio(2, 3),
                vec![
                    Status::Candidate,
                    Status::Elected,
                    Status::Candidate,
                    Status::Candidate,
                ],
                vec![FixedDecimal9::one(); 4]
            )
        );
        assert_eq!(
            count,
            VoteCount::new(
                [
                    FixedDecimal9::zero(),
                    FixedDecimal9::one(),
                    FixedDecimal9::zero(),
                    FixedDecimal9::zero(),
                ],
                FixedDecimal9::zero(),
            )
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"Action: Elect: Banana
	Elected:  Banana (1.000000000)
	Hopeful:  Apple (0.000000000)
	Hopeful:  Cherry (0.000000000)
	Hopeful:  Date (0.000000000)
	Quota: 0.333333334
	Votes: 1.000000000
	Residual: 0.000000000
	Total: 1.000000000
	Surplus: 0.000000000
"
        );

        // Second candidate is elected.
        let election = make_election([Ballot::new(3, [vec![1]]), Ballot::new(1, [vec![3]])]);
        let mut state = make_state(
            &election,
            [
                Status::Candidate,
                Status::Elected,
                Status::Candidate,
                Status::Candidate,
            ],
        );
        let mut count = make_count();

        let mut buf = Vec::new();
        let iteration = state.iterate_droop(&mut buf, &mut count).unwrap();
        assert_eq!(iteration, DroopIteration::Elected);
        assert_eq!(
            (
                state.threshold,
                state.surplus,
                state.statuses,
                state.keep_factors
            ),
            (
                FixedDecimal9::new(777_777_779),
                FixedDecimal9::new(777_777_777),
                vec![
                    Status::Candidate,
                    Status::Elected,
                    Status::Candidate,
                    Status::Elected,
                ],
                vec![
                    FixedDecimal9::one(),
                    FixedDecimal9::new(444_444_445),
                    FixedDecimal9::one(),
                    FixedDecimal9::one(),
                ]
            )
        );
        assert_eq!(
            count,
            VoteCount::new(
                [
                    FixedDecimal9::zero(),
                    FixedDecimal9::new(1_333_333_335),
                    FixedDecimal9::zero(),
                    FixedDecimal9::one(),
                ],
                FixedDecimal9::new(1_666_666_665),
            )
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"Action: Elect: Date
	Elected:  Banana (1.333333335)
	Elected:  Date (1.000000000)
	Hopeful:  Apple (0.000000000)
	Hopeful:  Cherry (0.000000000)
	Quota: 0.777777779
	Votes: 2.333333335
	Residual: 1.666666665
	Total: 4.000000000
	Surplus: 1.666666666
"
        );

        // Iteration stabilizes.
        let election = make_election([Ballot::new(1, [vec![1]])]);
        let mut state = make_state(
            &election,
            [
                Status::Candidate,
                Status::Elected,
                Status::Candidate,
                Status::Candidate,
            ],
        );
        let mut count = make_count();

        let mut buf = Vec::new();
        let iteration = state.iterate_droop(&mut buf, &mut count).unwrap();
        assert_eq!(iteration, DroopIteration::Stable);
        assert_eq!(
            (
                state.threshold,
                state.surplus,
                state.statuses,
                state.keep_factors
            ),
            (
                FixedDecimal9::new(17_438),
                FixedDecimal9::new(34_875),
                vec![
                    Status::Candidate,
                    Status::Elected,
                    Status::Candidate,
                    Status::Candidate,
                ],
                vec![
                    FixedDecimal9::one(),
                    FixedDecimal9::new(52_313),
                    FixedDecimal9::one(),
                    FixedDecimal9::one(),
                ]
            )
        );
        assert_eq!(
            count,
            VoteCount::new(
                [
                    FixedDecimal9::zero(),
                    FixedDecimal9::new(52_313),
                    FixedDecimal9::zero(),
                    FixedDecimal9::zero(),
                ],
                FixedDecimal9::new(999_947_687),
            )
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"	Stable state detected (0.000034875)
"
        );
    }

    #[test]
    fn test_elect_candidates() {
        let election = Election::builder()
            .title("Vegetable contest")
            .num_seats(4)
            .candidates([
                Candidate::new("apple", false),
                Candidate::new("banana", true),
                Candidate::new("cherry", false),
                Candidate::new("date", false),
                Candidate::new("eggplant", false),
                Candidate::new("fig", false),
                Candidate::new("grape", false),
            ])
            .build();
        let mut state = State::builder()
            .election(&election)
            .statuses([
                Status::Candidate,
                Status::Withdrawn,
                Status::Elected,
                Status::NotElected,
                Status::Candidate,
                Status::Candidate,
                Status::Candidate,
            ])
            .keep_factors([FixedDecimal9::zero(); 7])
            .threshold(FixedDecimal9::ratio(3, 2))
            .surplus(FixedDecimal9::zero())
            .omega(FixedDecimal9::zero())
            .parallel(Parallel::No)
            .force_positive_surplus(false)
            .pascal(None)
            .build();
        let count = VoteCount::new(
            [
                FixedDecimal9::from_usize(1),
                FixedDecimal9::zero(),
                FixedDecimal9::from_usize(2),
                FixedDecimal9::zero(),
                FixedDecimal9::from_usize(3),
                FixedDecimal9::new(1_499_999_999),
                FixedDecimal9::ratio(3, 2),
            ],
            FixedDecimal9::zero(),
        );

        let mut buf = Vec::new();
        state.elect_candidates(&mut buf, &count).unwrap();

        assert_eq!(
            state.statuses,
            vec![
                Status::Candidate,
                Status::Withdrawn,
                Status::Elected,
                Status::NotElected,
                Status::Elected,
                Status::Candidate,
                Status::Elected,
            ]
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"Action: Elect: Eggplant
	Elected:  Cherry (2.000000000)
	Elected:  Eggplant (3.000000000)
	Hopeful:  Apple (1.000000000)
	Hopeful:  Fig (1.499999999)
	Hopeful:  Grape (1.500000000)
	Defeated: Date (0.000000000)
	Quota: 1.500000000
	Votes: 8.999999999
	Residual: 0.000000000
	Total: 8.999999999
	Surplus: 0.000000000
Action: Elect: Grape
	Elected:  Cherry (2.000000000)
	Elected:  Eggplant (3.000000000)
	Elected:  Grape (1.500000000)
	Hopeful:  Apple (1.000000000)
	Hopeful:  Fig (1.499999999)
	Defeated: Date (0.000000000)
	Quota: 1.500000000
	Votes: 8.999999999
	Residual: 0.000000000
	Total: 8.999999999
	Surplus: 0.000000000
"
        );
    }

    #[test]
    fn test_elect_candidates_exact() {
        let election = Election::builder()
            .title("Vegetable contest")
            .num_seats(4)
            .candidates([
                Candidate::new("apple", false),
                Candidate::new("banana", true),
                Candidate::new("cherry", false),
                Candidate::new("date", false),
                Candidate::new("eggplant", false),
                Candidate::new("fig", false),
                Candidate::new("grape", false),
            ])
            .build();
        let mut state = State::builder()
            .election(&election)
            .statuses([
                Status::Candidate,
                Status::Withdrawn,
                Status::Elected,
                Status::NotElected,
                Status::Candidate,
                Status::Candidate,
                Status::Candidate,
            ])
            .keep_factors([
                BigRational::zero(),
                BigRational::zero(),
                BigRational::zero(),
                BigRational::zero(),
                BigRational::zero(),
                BigRational::zero(),
                BigRational::zero(),
            ])
            .threshold(BigRational::ratio(3, 2))
            .surplus(BigRational::zero())
            .omega(BigRational::zero())
            .parallel(Parallel::No)
            .force_positive_surplus(false)
            .pascal(None)
            .build();
        let count = VoteCount::new(
            [
                BigRational::from_usize(1),
                BigRational::zero(),
                BigRational::from_usize(2),
                BigRational::zero(),
                BigRational::from_usize(3),
                BigRational::ratio(1_499_999_999, 1_000_000_000),
                BigRational::ratio(3, 2),
            ],
            BigRational::zero(),
        );

        let mut buf = Vec::new();
        state.elect_candidates(&mut buf, &count).unwrap();

        assert_eq!(
            state.statuses,
            vec![
                Status::Candidate,
                Status::Withdrawn,
                Status::Elected,
                Status::NotElected,
                Status::Elected,
                Status::Candidate,
                Status::Candidate,
            ]
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"Action: Elect: Eggplant
	Elected:  Cherry (2)
	Elected:  Eggplant (3)
	Hopeful:  Apple (1)
	Hopeful:  Fig (1499999999/1000000000)
	Hopeful:  Grape (3/2)
	Defeated: Date (0)
	Quota: 3/2
	Votes: 8999999999/1000000000
	Residual: 0
	Total: 8999999999/1000000000
	Surplus: 0
"
        );
    }

    #[test]
    fn test_handle_remaining_candidates_elected() {
        let mut election = make_fake_election();
        election.num_seats = 4;
        let mut state = make_fake_state(&election);
        let mut count = make_fake_count();

        let mut buf = Vec::new();
        state
            .handle_remaining_candidates(&mut buf, &mut count)
            .unwrap();

        assert_eq!(
            state.statuses,
            vec![
                Status::Elected,
                Status::Withdrawn,
                Status::Elected,
                Status::NotElected,
                Status::Elected,
                Status::NotElected,
                Status::Elected,
                Status::NotElected,
                Status::NotElected,
            ]
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"Action: Elect remaining: Apple
	Elected:  Apple (0.777777777)
	Elected:  Cherry (1.666666666)
	Elected:  Grape (1.833333333)
	Hopeful:  Eggplant (0.875000000)
	Defeated: Date (0.100000000)
	Defeated: Hazelnut (0.200000000)
	Defeated: Fig, Jalapeno (0.000000000)
	Quota: 1.500000000
	Votes: 5.452777776
	Residual: 0.547222224
	Total: 6.000000000
	Surplus: 0.100000000
Action: Elect remaining: Eggplant
	Elected:  Apple (0.000000000)
	Elected:  Cherry (0.000000000)
	Elected:  Eggplant (0.000000000)
	Elected:  Grape (0.000000000)
	Defeated: Date, Fig, Hazelnut, Jalapeno (0.000000000)
	Quota: 1.500000000
	Votes: 0.000000000
	Residual: 0.000000000
	Total: 0.000000000
	Surplus: 0.100000000
"
        );
    }

    #[test]
    fn test_handle_remaining_candidates_defeated() {
        let mut election = make_fake_election();
        election.num_seats = 2;
        let mut state = make_fake_state(&election);
        let mut count = make_fake_count();

        let mut buf = Vec::new();
        state
            .handle_remaining_candidates(&mut buf, &mut count)
            .unwrap();

        assert_eq!(
            state.statuses,
            vec![
                Status::NotElected,
                Status::Withdrawn,
                Status::Elected,
                Status::NotElected,
                Status::NotElected,
                Status::NotElected,
                Status::Elected,
                Status::NotElected,
                Status::NotElected,
            ]
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"Action: Defeat remaining: Apple
	Elected:  Cherry (1.666666666)
	Elected:  Grape (1.833333333)
	Hopeful:  Eggplant (0.875000000)
	Defeated: Apple (0.777777777)
	Defeated: Date (0.100000000)
	Defeated: Hazelnut (0.200000000)
	Defeated: Fig, Jalapeno (0.000000000)
	Quota: 1.500000000
	Votes: 5.452777776
	Residual: 0.547222224
	Total: 6.000000000
	Surplus: 0.100000000
Action: Defeat remaining: Eggplant
	Elected:  Cherry (0.000000000)
	Elected:  Grape (0.000000000)
	Defeated: Apple, Date, Eggplant, Fig, Hazelnut, Jalapeno (0.000000000)
	Quota: 1.500000000
	Votes: 0.000000000
	Residual: 0.000000000
	Total: 0.000000000
	Surplus: 0.100000000
"
        );
    }

    #[test]
    fn test_next_defeated_candidate() {
        let election = Election::builder()
            .title("Vegetable contest")
            .num_seats(2)
            .candidates([
                Candidate::new("apple", false),
                Candidate::new("banana", false),
                Candidate::new("cherry", false),
                Candidate::new("date", false),
            ])
            .build();
        let mut state = State::builder()
            .election(&election)
            .statuses([
                Status::Elected,
                Status::Candidate,
                Status::NotElected,
                Status::Candidate,
            ])
            .keep_factors([FixedDecimal9::zero(); 4])
            .threshold(FixedDecimal9::zero())
            .surplus(FixedDecimal9::ratio(1, 10))
            .omega(FixedDecimal9::zero())
            .parallel(Parallel::No)
            .force_positive_surplus(false)
            .pascal(None)
            .build();

        // One defeated candidate.
        let count = VoteCount::new(
            [
                FixedDecimal9::from_usize(1),
                FixedDecimal9::ratio(3, 10),
                FixedDecimal9::zero(),
                FixedDecimal9::ratio(5, 10),
            ],
            FixedDecimal9::zero(),
        );

        let logger = ThreadLocalLogger::start();
        let mut buf = Vec::new();
        let next = state.next_defeated_candidate(&mut buf, &count).unwrap();

        assert_eq!(next, 1);
        logger.check_target_level_logs(
            "stv_rs::meek",
            Debug,
            r"Lowest vote: 0.300000000 ~ 0.3
Low threshold: 0.400000000 ~ 0.4
Low candidates: [1]
",
        );
        assert!(buf.is_empty());

        // Tie break.
        let count = VoteCount::new(
            [
                FixedDecimal9::from_usize(1),
                FixedDecimal9::ratio(3, 10),
                FixedDecimal9::zero(),
                FixedDecimal9::ratio(3, 10),
            ],
            FixedDecimal9::zero(),
        );

        let logger = ThreadLocalLogger::start();
        let mut buf = Vec::new();
        let next = state.next_defeated_candidate(&mut buf, &count).unwrap();

        assert_eq!(next, 1);
        logger.check_target_level_logs(
            "stv_rs::meek",
            Debug,
            r"Lowest vote: 0.300000000 ~ 0.3
Low threshold: 0.400000000 ~ 0.4
Low candidates: [1, 3]
",
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"Action: Break tie (defeat): [Banana, Date] -> Banana
	Quota: 0.000000000
	Votes: 1.600000000
	Residual: 0.000000000
	Total: 1.600000000
	Surplus: 0.100000000
"
        );
    }

    #[test]
    fn test_write_candidate_counts() {
        let election = make_fake_election();
        let state = make_fake_state(&election);
        let count = make_fake_count();

        let mut buf = Vec::new();
        state.write_candidate_counts(&mut buf, &count).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"	Elected:  Cherry (1.666666666)
	Elected:  Grape (1.833333333)
	Hopeful:  Apple (0.777777777)
	Hopeful:  Eggplant (0.875000000)
	Defeated: Date (0.100000000)
	Defeated: Hazelnut (0.200000000)
	Defeated: Fig, Jalapeno (0.000000000)
"
        );
    }

    #[test]
    fn test_debug_count() {
        let logger = ThreadLocalLogger::start();

        let election = make_fake_election();
        let state = make_fake_state(&election);
        let count = make_fake_count();
        state.debug_count(&count);

        logger.check_target_level_logs(
            "stv_rs::meek",
            Debug,
            r"Sums:
    [0] grape (Elected) ~ 1.833333333
    [1] cherry (Elected) ~ 1.666666666
    [2] eggplant (Candidate) ~ 0.875
    [3] apple (Candidate) ~ 0.777777777
    [4] hazelnut (NotElected) ~ 0.2
    [5] date (NotElected) ~ 0.1
    [6] jalapeno (NotElected) ~ 0
    [7] fig (NotElected) ~ 0
    [8] banana (Withdrawn) ~ 0
    @exhausted ~ 0.547222224
Sum = 5.452777776
Total = 6.000000000
Elected:
    [0] cherry
    [1] grape
Not elected:
    [8] date
    [7] fig
    [6] hazelnut
    [5] jalapeno
",
        );
    }
}
