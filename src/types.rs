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

#[cfg(test)]
use std::borrow::Borrow;

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
}

#[cfg(test)]
impl Election {
    pub(crate) fn builder() -> ElectionBuilder {
        ElectionBuilder::default()
    }
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct ElectionBuilder {
    title: Option<String>,
    num_seats: Option<usize>,
    num_ballots: Option<usize>,
    candidates: Vec<Candidate>,
    ballots: Vec<Ballot>,
}

#[cfg(test)]
impl ElectionBuilder {
    pub(crate) fn build(self) -> Election {
        let num_ballots = self
            .num_ballots
            .unwrap_or_else(|| self.ballots.iter().map(|b| b.count).sum());
        Election {
            title: self.title.unwrap(),
            num_candidates: self.candidates.len(),
            num_seats: self.num_seats.unwrap(),
            num_ballots,
            candidates: self.candidates,
            ballots: self.ballots,
        }
    }

    pub(crate) fn title(mut self, title: &str) -> Self {
        self.title = Some(title.to_owned());
        self
    }

    pub(crate) fn num_seats(mut self, num_seats: usize) -> Self {
        self.num_seats = Some(num_seats);
        self
    }

    pub(crate) fn num_ballots(mut self, num_ballots: usize) -> Self {
        self.num_ballots = Some(num_ballots);
        self
    }

    pub(crate) fn check_num_ballots(mut self, num_ballots: usize) -> Self {
        assert_eq!(num_ballots, self.ballots.iter().map(|b| b.count).sum());
        self.num_ballots = Some(num_ballots);
        self
    }

    pub(crate) fn candidates(mut self, candidates: impl Borrow<[Candidate]>) -> Self {
        self.candidates = candidates.borrow().to_owned();
        self
    }

    pub(crate) fn ballots(mut self, ballots: impl Borrow<[Ballot]>) -> Self {
        self.ballots = ballots.borrow().to_owned();
        self
    }
}

/// Candidate in an election.
#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(test, derive(Clone))]
pub struct Candidate {
    /// Nickname, used for parsing ballots.
    pub nickname: String,
    /// Full name, used to output results.
    pub name: String,
    /// Whether the candidate has withdrawn from the election.
    pub is_withdrawn: bool,
}

#[cfg(test)]
impl Candidate {
    pub(crate) fn new(nickname: &str, is_withdrawn: bool) -> Self {
        let mut name = nickname.as_bytes().to_owned();
        if let Some(x) = name.first_mut() {
            *x = x.to_ascii_uppercase();
        }
        let name = String::from_utf8(name).unwrap();
        Candidate {
            nickname: nickname.to_owned(),
            name,
            is_withdrawn,
        }
    }
}

/// Ballot cast in the election.
#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(test, derive(Clone))]
pub struct Ballot {
    /// Number of electors that have cast this ballot.
    pub count: usize,
    /// Ordering of candidates in this ballot. The outer [`Vec`] represents the
    /// ranking of candidates, from most preferred to least preferred. The inner
    /// [`Vec`] represents candidates ranked equally at a given order.
    pub order: Vec<Vec<usize>>,
}

#[cfg(test)]
impl Ballot {
    pub(crate) fn new(count: usize, order: impl Borrow<[Vec<usize>]>) -> Self {
        Ballot {
            count,
            order: order.borrow().to_owned(),
        }
    }
}

impl Ballot {
    /// Checks that a ballot is valid, i.e. that no candidate appears twice in
    /// the ballot.
    pub fn validate(&self) {
        assert!(self.count > 0);
        let mut all: Vec<usize> = self.candidates().collect();
        all.sort_unstable();
        let len = all.len();
        all.dedup();
        assert_eq!(len, all.len());
    }

    /// Returns the set of candidates present in this ballot.
    fn candidates(&self) -> impl Iterator<Item = usize> + '_ {
        self.order.iter().flatten().cloned()
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
