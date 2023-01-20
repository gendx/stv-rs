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

/// Election input, representing a parsed ballot file.
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

/// Candidate in an election.
pub struct Candidate {
    /// Nickname, used for parsing ballots.
    pub nickname: String,
    /// Full name, used to output results.
    pub name: String,
    /// Whether the candidate has withdrawn from the election.
    pub is_withdrawn: bool,
}

/// Ballot cast in the election.
pub struct Ballot {
    /// Number of electors that have cast this ballot.
    pub count: usize,
    /// Ordering of candidates in this ballot. The outer [`Vec`] represents the
    /// ranking of candidates, from most preferred to least preferred. The inner
    /// [`Vec`] represents candidates ranked equally at a given order.
    pub order: Vec<Vec<usize>>,
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
pub struct ElectionResult {
    /// List of elected candidates, by election order.
    pub elected: Vec<usize>,
    /// List of non-elected candidates, by non-election order.
    pub not_elected: Vec<usize>,
}
