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
use std::io;
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

#[cfg(test)]
impl<'e, I, R> State<'e, I, R> {
    fn builder() -> test::StateBuilder<'e, I, R> {
        test::StateBuilder::default()
    }
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
    fn new(election: &Election, omega_exponent: usize, parallel: bool) -> State<'_, I, R> {
        State {
            election,
            statuses: election
                .candidates
                .iter()
                .map(|c| {
                    if c.is_withdrawn {
                        Status::Withdrawn
                    } else {
                        Status::Candidate
                    }
                })
                .collect(),
            elected: Vec::new(),
            not_elected: Vec::new(),
            to_elect: election.num_seats,
            keep_factors: vec![R::one(); election.candidates.len()],
            threshold: VoteCount::<I, R>::threshold_exhausted(election, &R::zero()),
            surplus: R::zero(),
            omega: R::one() / &(0..omega_exponent).map(|_| R::from_usize(10)).product(),
            parallel,
            _phantom: PhantomData,
        }
    }

    /// Runs an election according to Meek's rules. This aims to produce
    /// reproducible results w.r.t. Droop.py.
    pub fn stv_droop(
        stdout: &mut impl io::Write,
        election: &Election,
        package_name: &str,
        omega_exponent: usize,
        parallel: bool,
    ) -> io::Result<ElectionResult> {
        info!(
            "Parallel vote counting is {}",
            if parallel { "enabled" } else { "disabled" }
        );

        let mut state = Self::new(election, omega_exponent, parallel);

        let beginning = Instant::now();
        let mut timestamp = beginning;

        let mut count = state.start_election(stdout, package_name, omega_exponent)?;

        for round in 1.. {
            if state.election_completed(stdout, round)? {
                break;
            }

            state.election_round(stdout, &mut count, round)?;

            let now = Instant::now();
            debug!(
                "Elapsed time to compute this round: {:?} / total: {:?}",
                now.duration_since(timestamp),
                now.duration_since(beginning)
            );
            timestamp = now;
        }

        state.handle_remaining_candidates(stdout, &mut count)?;

        state.write_action(stdout, "Count Complete", &count, true)?;

        let now = Instant::now();
        info!("Total elapsed time: {:?}", now.duration_since(beginning));

        let result = ElectionResult {
            elected: state.elected,
            not_elected: state.not_elected,
        };

        writeln!(stdout)?;

        Ok(result)
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

        debug!("Weights:");
        for (i, candidate) in self.election.candidates.iter().enumerate() {
            debug!(
                "    [{i}] {} ({:?}) ~ {}",
                candidate.nickname,
                self.statuses[i],
                self.keep_factors[i].to_f64()
            );
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
        round: usize,
    ) -> io::Result<()> {
        let iteration = self.iterate_droop(stdout, count, round)?;
        self.write_action(stdout, &format!("Iterate ({iteration})"), count, false)?;

        if let DroopIteration::Elected = iteration {
            debug!("Iteration returned Elected, continuing the loop");
        } else {
            let not_elected = self.next_defeated_candidate(stdout, count)?;

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
            self.write_action(stdout, &message, count, true)?;

            self.keep_factors[not_elected] = R::zero();
            count.sum[not_elected] = R::zero();

            *count = self.count_votes();
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
        round: usize,
    ) -> io::Result<DroopIteration> {
        let mut last_surplus = R::from_usize(self.election.num_ballots);

        loop {
            *count = self.count_votes();
            self.threshold = count.threshold(self.election);

            let has_elected = self.elect_candidates(stdout, count, round)?;

            self.surplus = count.surplus(&self.threshold, &self.elected);

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
                let mut new_factor = self.keep_factors[i]
                    .mul_up(&self.threshold)
                    .div_up(&count.sum[i]);
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

    /// Elect candidates that exceed the threshold, returning true if any
    /// candidate was elected.
    fn elect_candidates(
        &mut self,
        stdout: &mut impl io::Write,
        count: &VoteCount<I, R>,
        round: usize,
    ) -> io::Result<bool> {
        let mut has_elected = false;
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
                    self.elect_candidate(i);
                    self.write_action(
                        stdout,
                        &format!("Elect remaining: {}", candidate.name),
                        count,
                        true,
                    )?;
                } else {
                    self.defeat_candidate(i);
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
        &self,
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
            writeln!(
                stdout,
                "\t{status} {} ({})",
                self.election.candidates[i].name, count.sum[i]
            )?;
        }

        // Candidates beyond the first defeated one with a zero count must all be
        // defeated and with a zero count.
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
    use crate::arithmetic::fixed::FixedDecimal9;
    use crate::types::{Ballot, Candidate};
    use crate::util::log_tester::ThreadLocalLogger;
    use log::Level::{self, Debug, Info};
    use num::traits::{One, Zero};
    use num::BigRational;

    pub struct StateBuilder<'e, I, R> {
        election: Option<&'e Election>,
        statuses: Option<Vec<Status>>,
        keep_factors: Option<Vec<R>>,
        threshold: Option<R>,
        surplus: Option<R>,
        omega: Option<R>,
        parallel: Option<bool>,
        _phantom: PhantomData<I>,
    }

    impl<'e, I, R> Default for StateBuilder<'e, I, R> {
        fn default() -> Self {
            StateBuilder {
                election: None,
                statuses: None,
                keep_factors: None,
                threshold: None,
                surplus: None,
                omega: None,
                parallel: None,
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

        fn parallel(mut self, parallel: bool) -> Self {
            self.parallel = Some(parallel);
            self
        }

        fn build(self) -> State<'e, I, R> {
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
            State {
                election,
                statuses,
                elected,
                not_elected,
                to_elect,
                keep_factors: self.keep_factors.unwrap(),
                threshold: self.threshold.unwrap(),
                surplus: self.surplus.unwrap(),
                omega: self.omega.unwrap(),
                parallel: self.parallel.unwrap(),
                _phantom: PhantomData,
            }
        }
    }

    #[track_caller]
    fn check_logs_debug(logger: ThreadLocalLogger, expected: &str) {
        let mut report = String::new();
        for record in logger.into_iter() {
            assert_eq!(record.target, "stv_rs::meek");
            assert_eq!(record.level, Debug);
            report.push_str(&record.message);
            report.push('\n');
        }

        assert_eq!(report, expected);
    }

    #[track_caller]
    fn check_logs<'a>(
        logger: ThreadLocalLogger,
        expected: impl IntoIterator<Item = (Level, &'a str)>,
    ) {
        let mut report = Vec::new();
        for record in logger.into_iter() {
            assert_eq!(record.target, "stv_rs::meek");
            report.push((record.level, record.message));
        }

        let expected_report = expected
            .into_iter()
            .map(|(level, msg)| (level, msg.to_owned()))
            .collect::<Vec<_>>();

        assert_eq!(report, expected_report);
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

    fn make_fake_state(election: &Election) -> State<i64, FixedDecimal9> {
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
            .parallel(false)
            .build()
    }

    fn make_fake_count() -> VoteCount<i64, FixedDecimal9> {
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
    fn test_stv_droop() {
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
        let result =
            State::<i64, FixedDecimal9>::stv_droop(&mut buf, &election, "package name", 6, false)
                .unwrap();
        assert_eq!(
            result,
            ElectionResult {
                elected: vec![6, 7, 8, 4, 3],
                not_elected: vec![0, 2]
            }
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
        let state = State::new(&election, omega_exponent, false);

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

        fn make_state(election: &Election, statuses: [Status; 5]) -> State<'_, i64, FixedDecimal9> {
            State::builder()
                .election(election)
                .statuses(statuses)
                .keep_factors([FixedDecimal9::zero(); 5])
                .threshold(FixedDecimal9::zero())
                .surplus(FixedDecimal9::zero())
                .omega(FixedDecimal9::zero())
                .parallel(false)
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
        check_logs(
            logger,
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
        check_logs(
            logger,
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
        check_logs(
            logger,
            [
                (
                    Debug,
                    "Checking if count is complete: candidates=2, to_elect=1",
                ),
                (Debug, "Weights:"),
                (Debug, "    [0] apple (Candidate) ~ 0"),
                (Debug, "    [1] banana (Withdrawn) ~ 0"),
                (Debug, "    [2] cherry (Elected) ~ 0"),
                (Debug, "    [3] date (NotElected) ~ 0"),
                (Debug, "    [4] eggplant (Candidate) ~ 0"),
            ],
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
        ) -> State<'_, i64, FixedDecimal9> {
            State::builder()
                .election(election)
                .statuses(statuses)
                .keep_factors([FixedDecimal9::one(); 4])
                .threshold(FixedDecimal9::zero())
                .surplus(FixedDecimal9::zero())
                .omega(FixedDecimal9::ratio(1, 1_000_000))
                .parallel(false)
                .build()
        }

        fn make_count() -> VoteCount<i64, FixedDecimal9> {
            VoteCount::new([FixedDecimal9::zero(); 4], FixedDecimal9::zero())
        }

        // No ballot.
        let election = make_election([]);
        let mut state = make_state(&election, [Status::Candidate; 4]);
        let mut count = make_count();

        let mut buf = Vec::new();
        state.election_round(&mut buf, &mut count, 42).unwrap();
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
        state.election_round(&mut buf, &mut count, 42).unwrap();
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
        state.election_round(&mut buf, &mut count, 42).unwrap();
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
        state.election_round(&mut buf, &mut count, 42).unwrap();
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
        ) -> State<'_, i64, FixedDecimal9> {
            State::builder()
                .election(election)
                .statuses(statuses)
                .keep_factors([FixedDecimal9::one(); 4])
                .threshold(FixedDecimal9::zero())
                .surplus(FixedDecimal9::zero())
                .omega(FixedDecimal9::ratio(1, 1_000_000))
                .parallel(false)
                .build()
        }

        fn make_count() -> VoteCount<i64, FixedDecimal9> {
            VoteCount::new([FixedDecimal9::zero(); 4], FixedDecimal9::zero())
        }

        // No ballot.
        let election = make_election([]);
        let mut state = make_state(&election, [Status::Candidate; 4]);
        let mut count = make_count();

        let mut buf = Vec::new();
        let iteration = state.iterate_droop(&mut buf, &mut count, 42).unwrap();
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
        let iteration = state.iterate_droop(&mut buf, &mut count, 42).unwrap();
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
        let iteration = state.iterate_droop(&mut buf, &mut count, 42).unwrap();
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
        let iteration = state.iterate_droop(&mut buf, &mut count, 42).unwrap();
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
            .parallel(false)
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
        state.elect_candidates(&mut buf, &count, 42).unwrap();

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
            .parallel(false)
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
        state.elect_candidates(&mut buf, &count, 42).unwrap();

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
        let state = State::builder()
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
            .parallel(false)
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
        check_logs_debug(
            logger,
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
        check_logs_debug(
            logger,
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

        check_logs_debug(
            logger,
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
