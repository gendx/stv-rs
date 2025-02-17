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

//! Types to represent an election.

mod ballot;
mod util;

pub use ballot::Ballot;
use log::Level::{Debug, Trace};
use log::{debug, log_enabled, trace};
use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap};
use util::count_vec_allocations;

/// Election input, representing a parsed ballot file.
#[derive(Debug, PartialEq, Eq)]
pub struct Election {
    /// Name of the election.
    pub title: String,
    /// Number of candidates.
    pub num_candidates: usize,
    /// Number of elected seats.
    pub num_seats: usize,
    /// Number of ballots that were cast in the election.
    pub num_ballots: usize,
    /// Candidates in this election.
    pub candidates: Vec<Candidate>,
    /// Ballots that were cast in this election.
    pub ballots: Vec<Ballot>,
    /// Tie-break order of candidates, mapping each candidate ID to its order
    /// in the tie break.
    pub tie_order: HashMap<usize, usize>,
}

impl Election {
    /// Returns a new builder.
    pub fn builder() -> ElectionBuilder {
        ElectionBuilder::default()
    }

    /// Returns true if any ballot contains candidates ranked equally.
    pub fn has_any_tied_ballot(&self) -> bool {
        self.ballots.iter().any(|b| b.has_tie())
    }

    pub(crate) fn debug_allocations(&self) {
        if !log_enabled!(Debug) {
            return;
        }

        let mut allocations = BTreeMap::new();
        count_vec_allocations(&mut allocations, &self.ballots);
        for b in &self.ballots {
            b.count_allocations(&mut allocations);
        }
        let mut total_count = 0;
        let mut total_size = 0;
        for (size, count) in allocations.iter() {
            total_count += count;
            total_size += size * count;
            debug!(
                "Allocations of {size} bytes: {count} => {} bytes",
                size * count
            );
        }
        debug!("Ballots use {total_size} bytes in {total_count} allocations");
        let ballots_len = self.ballots.len() as f64;
        debug!(
            "Each ballot uses {} bytes in {} allocations",
            total_size as f64 / ballots_len,
            total_count as f64 / ballots_len
        );

        if !log_enabled!(Trace) {
            return;
        }

        trace!("Allocated addresses:");
        let mut diffs = BTreeMap::new();
        let mut prev = None;
        for b in &self.ballots {
            for address in b.allocated_addresses() {
                match prev {
                    None => trace!("- {address:#x?}"),
                    Some(prev) => {
                        let diff = if address >= prev {
                            trace!("- {address:#x?} (+{:#x?})", address - prev);
                            address - prev
                        } else {
                            trace!("- {address:#x?} (-{:#x?})", prev - address);
                            prev - address
                        };
                        *diffs.entry(diff.checked_ilog2().unwrap_or(0)).or_insert(0) += 1;
                    }
                }
                prev = Some(address);
            }
        }

        trace!("Histogram of sequential jumps between allocated addresses:");
        let max_count = diffs.values().max().unwrap();
        for (&diff_log2, count) in diffs.iter() {
            let start = if diff_log2 == 0 { 0 } else { 1u64 << diff_log2 };
            let end = (1u64 << (diff_log2 + 1)) - 1;

            let count_stars = count * 40 / max_count;
            let count_spaces = 40 - count_stars;
            let stars = &"****************************************"[..count_stars];
            let spaces = &"                                        "[..count_spaces];
            trace!("{stars}{spaces} {start}..={end}: {count}");
        }
    }
}

/// Builder for the [`Election`] type.
#[derive(Default)]
pub struct ElectionBuilder {
    title: Option<String>,
    num_seats: Option<usize>,
    num_ballots: Option<usize>,
    candidates: Vec<Candidate>,
    ballots: Vec<Ballot>,
    tie_order: Option<HashMap<usize, usize>>,
}

impl ElectionBuilder {
    /// Build the [`Election`] object.
    pub fn build(self) -> Election {
        let num_ballots = self
            .num_ballots
            .unwrap_or_else(|| self.ballots.iter().map(|b| b.count()).sum());
        let num_candidates = self.candidates.len();
        Election {
            title: self.title.unwrap(),
            num_candidates,
            num_seats: self.num_seats.unwrap(),
            num_ballots,
            candidates: self.candidates,
            ballots: self.ballots,
            tie_order: self
                .tie_order
                .unwrap_or_else(|| (0..num_candidates).map(|i| (i, i)).collect()),
        }
    }

    /// Sets the name of the election.
    pub fn title(mut self, title: &str) -> Self {
        self.title = Some(title.to_owned());
        self
    }

    /// Sets the number of elected seats.
    pub fn num_seats(mut self, num_seats: usize) -> Self {
        self.num_seats = Some(num_seats);
        self
    }

    /// Sets the number of ballots cast in the election.
    pub fn num_ballots(mut self, num_ballots: usize) -> Self {
        self.num_ballots = Some(num_ballots);
        self
    }

    /// Checks that the given number of ballots is consistent with the actual
    /// number of ballots previously set with [`Self::ballots()`].
    pub fn check_num_ballots(mut self, num_ballots: usize) -> Self {
        assert_eq!(num_ballots, self.ballots.iter().map(|b| b.count()).sum());
        self.num_ballots = Some(num_ballots);
        self
    }

    /// Sets the list of candidates in the election.
    pub fn candidates(mut self, candidates: impl Into<Vec<Candidate>>) -> Self {
        self.candidates = candidates.into();
        self
    }

    /// Sets the list of ballots in the election.
    pub fn ballots(mut self, ballots: impl Into<Vec<Ballot>>) -> Self {
        self.ballots = ballots.into();
        self
    }

    /// Sets the tie-break order of candidates in the election.
    pub fn tie_order(mut self, order: impl Borrow<[usize]>) -> Self {
        assert_eq!(order.borrow().len(), self.candidates.len());
        let mut tie_order = HashMap::new();
        for (i, &c) in order.borrow().iter().enumerate() {
            assert!(c < self.candidates.len());
            assert!(tie_order.insert(c, i).is_none());
        }
        self.tie_order = Some(tie_order);
        self
    }
}

/// Candidate in an election.
#[derive(Debug, PartialEq, Eq)]
pub struct Candidate {
    /// Nickname, used for parsing ballots.
    pub nickname: String,
    /// Full name, used to output results.
    pub name: String,
    /// Whether the candidate has withdrawn from the election.
    pub is_withdrawn: bool,
}

impl Candidate {
    /// Constructs a new [`Candidate`].
    pub fn new(nickname: impl Into<String>, is_withdrawn: bool) -> Self {
        let nickname = nickname.into();
        let mut name = nickname.clone().into_bytes();
        if let Some(x) = name.first_mut() {
            *x = x.to_ascii_uppercase();
        }
        let name = String::from_utf8(name).unwrap();
        Candidate {
            nickname,
            name,
            is_withdrawn,
        }
    }
}

/// An election result.
#[derive(Debug, PartialEq, Eq)]
pub struct ElectionResult {
    /// List of elected candidates, by election order.
    pub elected: Vec<usize>,
    /// List of non-elected candidates, by non-election order.
    pub not_elected: Vec<usize>,
}
