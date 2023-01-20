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

use crate::arithmetic::{Integer, Rational};
use crate::types::{Election, ElectionResult};
use crate::vote_count::VoteCount;
use log::{debug, info, warn};
use std::fmt::{self, Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::time::Instant;

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
pub struct State<'e, I, R> {
    /// Election input.
    election: &'e Election,
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
    parallel: bool,
    _phantom: PhantomData<I>,
}

impl<I, R> State<'_, I, R>
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
    /// Runs an election according to Meek's rules. This aims to produce
    /// reproducible results w.r.t. Droop.py.
    pub fn stv_droop(
        election: &Election,
        package_name: &str,
        omega_exponent: usize,
        parallel: bool,
    ) -> ElectionResult {
        info!("Parallel vote counting is {}", if parallel { "enabled" } else { "disabled" });

        let mut state = State {
            election,
            statuses: election
                .candidates
                .iter()
                .map(|c| if c.is_withdrawn { Status::Withdrawn } else { Status::Candidate })
                .collect(),
            elected: Vec::new(),
            not_elected: Vec::new(),
            to_elect: election.num_seats,
            keep_factors: vec![R::one(); election.candidates.len()],
            threshold: VoteCount::<I, R>::threshold_exhausted(election, &R::zero()),
            surplus: R::zero(),
            omega: R::one()
                / &(0..omega_exponent).map(|_| R::from_int(I::from_usize(10))).product(),
            parallel,
            _phantom: PhantomData,
        };

        let beginning = Instant::now();
        let mut timestamp = beginning;

        let mut count = state.start_election(package_name, omega_exponent);

        for round in 1.. {
            if state.election_completed(round) {
                break;
            }

            state.election_round(&mut count, round);

            let now = Instant::now();
            debug!(
                "Elapsed time to compute this round: {:?} / total: {:?}",
                now.duration_since(timestamp),
                now.duration_since(beginning)
            );
            timestamp = now;
        }

        state.handle_remaining_candidates(&mut count);

        state.print_action("Count Complete", &count, true);

        let now = Instant::now();
        info!("Total elapsed time: {:?}", now.duration_since(beginning));

        let result = ElectionResult { elected: state.elected, not_elected: state.not_elected };

        println!();

        result
    }

    /// Elects a candidate, and updates the state accordingly.
    fn elect_candidate(&mut self, i: usize) {
        info!("Electing candidate {i}");
        assert!(self.to_elect != 0);
        self.elected.push(i);
        self.statuses[i] = Status::Elected;
        self.to_elect -= 1;
    }

    /// Defeats a candidate, and updates the state accordingly.
    fn defeat_candidate(&mut self, i: usize) {
        info!("Defeating candidate {i}");
        self.not_elected.push(i);
        self.statuses[i] = Status::NotElected;
    }

    /// Returns the number of candidates.
    fn num_candidates(&self) -> usize {
        self.election.num_candidates
    }

    /// Counts the ballots based on the current keep factors.
    fn count_votes(&self) -> VoteCount<I, R> {
        VoteCount::<I, R>::count_votes(self.election, &self.keep_factors, self.parallel)
    }

    /// Performs the initial counting to start the election.
    fn start_election(&self, package_name: &str, omega_exponent: usize) -> VoteCount<I, R> {
        println!(
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
        );

        for candidate in &self.election.candidates {
            println!(
                "\tAdd {}: {}",
                if candidate.is_withdrawn { "withdrawn" } else { "eligible" },
                candidate.name
            );
        }

        let mut count = self.count_votes();
        // Droop somehow doesn't report exhausted count on the initial count.
        count.exhausted = R::zero();

        self.print_action("Begin Count", &count, true);

        count
    }

    /// Returns true if the election is complete.
    fn election_completed(&self, round: usize) -> bool {
        let candidate_count =
            self.statuses.iter().filter(|&&status| status == Status::Candidate).count();
        debug!(
            "Checking if count is complete: candidates={candidate_count}, to_elect={}",
            self.to_elect
        );
        assert!(candidate_count >= self.to_elect);
        if self.to_elect == 0 || candidate_count == self.to_elect {
            info!("Count is now complete!");
            return true;
        }
        println!("Round {round}:");

        debug!("Weights:");
        for (i, candidate) in self.election.candidates.iter().enumerate() {
            debug!(
                "    [{i}] {} ({:?}) ~ {}",
                candidate.nickname,
                self.statuses[i],
                self.keep_factors[i].to_f64()
            );
        }

        if self.statuses.iter().all(|&status| status != Status::Candidate) {
            debug!("Election done!");
            return true;
        }

        false
    }

    /// Runs one election round, which either elects or defeats exactly one
    /// candidate.
    fn election_round(&mut self, count: &mut VoteCount<I, R>, round: usize) {
        let iteration = self.iterate_droop(count, round);
        self.print_action(&format!("Iterate ({iteration})"), count, false);

        if let DroopIteration::Elected = iteration {
            debug!("Iteration returned Elected, continuing the loop");
        } else {
            let not_elected = self.next_defeated_candidate(count);

            self.defeat_candidate(not_elected);
            let message = match iteration {
                DroopIteration::Stable => {
                    format!(
                        "Defeat (stable surplus {}): {}",
                        self.surplus, self.election.candidates[not_elected].name
                    )
                }
                DroopIteration::Omega => {
                    format!(
                        "Defeat (surplus {} < omega): {}",
                        self.surplus, self.election.candidates[not_elected].name
                    )
                }
                DroopIteration::Elected => unreachable!(),
            };
            self.print_action(&message, count, true);

            self.keep_factors[not_elected] = R::zero();
            count.sum[not_elected] = R::zero();

            *count = self.count_votes();
        }

        self.debug_count(count);
    }

    /// Iterates the vote counting process until either a candidate is elected,
    /// the surplus becomes lower than omega, or the count has stabilized.
    fn iterate_droop(&mut self, count: &mut VoteCount<I, R>, round: usize) -> DroopIteration {
        let mut status = None;
        let mut last_surplus = R::from_int(I::from_usize(self.election.num_ballots));

        loop {
            *count = self.count_votes();
            self.threshold = count.threshold(self.election);

            // Elect candidates that exceed the threshold.
            for (i, candidate) in self.election.candidates.iter().enumerate() {
                match self.statuses[i] {
                    // Elected candidates must always keep at least the threshold of ballots.
                    // Otherwise their keep factor was too low, i.e. too much of
                    // their ballots were unduly transfered to other candidates.
                    Status::Elected => {
                        if count.sum[i] < self.threshold {
                            warn!(
                                "Count for elected candidate {} is lower than the quota: {} < {} ~ {} < {}",
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
                            info!("Elected in round {round}: {}", candidate.nickname);
                            self.elect_candidate(i);
                            self.print_action(&format!("Elect: {}", candidate.name), count, true);
                            status = Some(DroopIteration::Elected);
                        }
                    }
                    Status::Withdrawn => continue,
                }
            }

            self.surplus = count.surplus(&self.threshold, &self.elected);

            if let Some(DroopIteration::Elected) = status {
                debug!("Returning from iterate_droop (Elected)");
                return DroopIteration::Elected;
            }

            if self.surplus <= self.omega {
                debug!("Returning from iterate_droop (Omega)");
                return DroopIteration::Omega;
            }

            if self.surplus >= last_surplus {
                debug!("Returning from iterate_droop (Stable)");
                return DroopIteration::Stable;
            }

            last_surplus = self.surplus.clone();

            // Update keep factors of elected candidates.
            debug!("Updating keep factors");
            for i in
                self.statuses.iter().enumerate().filter_map(|(i, &status)| {
                    if status == Status::Elected { Some(i) } else { None }
                })
            {
                let mut new_factor =
                    self.keep_factors[i].mul_up(&self.threshold).div_up(&count.sum[i]);
                new_factor.ceil_precision();
                debug!(
                    "\t{}: {} ~ {} -> {new_factor} ~ {}",
                    self.election.candidates[i].nickname,
                    self.keep_factors[i],
                    self.keep_factors[i].to_f64(),
                    new_factor.to_f64()
                );
                self.keep_factors[i] = new_factor;
            }
        }
    }

    /// Elects or defeat all remaining candidates, once a quorum is
    /// defeated or elected (respectively).
    fn handle_remaining_candidates(&mut self, count: &mut VoteCount<I, R>) {
        debug!("Checking remaining candidates");
        for (i, candidate) in self.election.candidates.iter().enumerate() {
            if self.statuses[i] == Status::Candidate {
                if self.elected.len() < self.election.num_seats {
                    self.elect_candidate(i);
                    self.print_action(&format!("Elect remaining: {}", candidate.name), count, true);
                } else {
                    self.defeat_candidate(i);
                    self.print_action(
                        &format!("Defeat remaining: {}", candidate.name),
                        count,
                        true,
                    );
                    self.keep_factors[i] = R::zero();
                    count.sum[i] = R::zero();
                }
                *count = self.count_votes();
            }
        }
    }

    /// Returns the next candidate to defeat.
    fn next_defeated_candidate(&self, count: &VoteCount<I, R>) -> usize {
        let min_sum = count
            .sum
            .iter()
            .enumerate()
            .filter_map(|(i, x)| if self.statuses[i] == Status::Candidate { Some(x) } else { None })
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        debug!("Lowest vote: {min_sum} ~ {}", min_sum.to_f64());

        let low_threshold = min_sum + &self.surplus;
        debug!("Low threshold: {low_threshold} ~ {}", low_threshold.to_f64());

        let low_candidates = (0..self.num_candidates())
            .filter(|&i| self.statuses[i] == Status::Candidate && count.sum[i] <= low_threshold)
            .collect::<Vec<_>>();
        debug!("Low candidates: {low_candidates:?}");

        // TODO: Break ties.
        assert_eq!(low_candidates.len(), 1);

        low_candidates[0]
    }

    /// Prints an action to stdout.
    fn print_action(&self, action: &str, count: &VoteCount<I, R>, print_full_count: bool) {
        println!("Action: {action}");
        if print_full_count {
            self.print_candidate_counts(count);
        }
        count.print_stats(&self.threshold, &self.surplus);
    }

    /// Prints current candidate counts to stdout.
    fn print_candidate_counts(&self, count: &VoteCount<I, R>) {
        // Sort candidates: elected first, then candidates and lastly non-elected.
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
            println!("\t{status} {} ({})", self.election.candidates[i].name, count.sum[i]);
        }

        // Candidates beyond the first defeated one with a zero count must all be
        // defeated and with a zero count.
        if let Some(split) = split {
            print!("\tDefeated: ");
            for (rank, &i) in droop_sorted_candidates[split..].iter().enumerate() {
                assert_eq!(self.statuses[i], Status::NotElected);
                assert_eq!(count.sum[i], R::zero());
                if rank != 0 {
                    print!(", ");
                }
                print!("{}", self.election.candidates[i].name);
            }
            println!(" ({})", R::zero());
        }
    }

    /// Displays a debug output of the current vote counts.
    fn debug_count(&self, count: &VoteCount<I, R>) {
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
            R::from_int(I::from_usize(self.election.num_ballots)),
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
