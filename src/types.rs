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

use std::borrow::Borrow;
use std::collections::HashMap;

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
            .unwrap_or_else(|| self.ballots.iter().map(|b| b.count).sum());
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
        assert_eq!(num_ballots, self.ballots.iter().map(|b| b.count).sum());
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

/// Ballot cast in the election.
#[derive(Debug, PartialEq, Eq)]
pub struct Ballot {
    /// Number of electors that have cast this ballot.
    pub count: usize,
    /// Ordering of candidates in this ballot. The outer [`Vec`] represents the
    /// ranking of candidates, from most preferred to least preferred. The inner
    /// [`Vec`] represents candidates ranked equally at a given order.
    pub order: Vec<Vec<usize>>,
}

impl Ballot {
    /// Constructs a new [`Ballot`].
    pub fn new(count: usize, order: impl Into<Vec<Vec<usize>>>) -> Self {
        Ballot {
            count,
            order: order.into(),
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
