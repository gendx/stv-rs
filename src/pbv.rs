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

//! A simulation of plurality block voting based on ranked ballots.

use crate::arithmetic::{Integer, IntegerRef, Rational, RationalRef};
use crate::types::{BallotView, Election, ElectionResult};
use log::{debug, info, trace};
use std::io;

/// A simulation of plurality block voting, where each ballot attributes 1 vote
/// to the first N candidates (where N is the number of seats), and 0 votes to
/// the following ones.
///
/// More precisely:
/// - If the ballot strictly ranks at least N candidates, the first N candidates
///   in the ballot receive 1 vote each.
/// - If the ballot ranks less than N candidates, all the ranked candidates
///   receive 1 vote each.
/// - If the ballot is structured as X candidates ranked, then K candidates
///   ranked equally, then possibly more candidates ranked, with `X < N < X +
///   K`, then the first X candidates receive 1 vote each, and the next K
///   candidates receive `(N - X) / K` votes each (i.e. the remaining `N - X`
///   votes are split equally among the K candidates ranked equally).
pub fn plurality_block_voting<I, R>(
    stdout: &mut impl io::Write,
    election: &Election,
    package_name: &str,
    votes_per_ballot: Option<usize>,
) -> io::Result<ElectionResult>
where
    I: Integer + Send + Sync,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I> + Send + Sync,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    let votes_per_ballot = votes_per_ballot.unwrap_or(election.num_seats);
    info!(
        "Plurality block voting with {} seats / {} candidates, votes per ballot = {votes_per_ballot}",
        election.num_seats, election.num_candidates
    );

    writeln!(
        stdout,
        r"
Election: {}

	{package_name}
	Rule: Simulated Plurality Block Voting
	Arithmetic: {}
	Seats: {}
	Ballots: {}
	Votes per ballot: {votes_per_ballot}
",
        election.title,
        R::description(),
        election.num_seats,
        election.num_ballots,
    )?;

    let mut sum = vec![R::zero(); election.num_candidates];
    for (i, ballot) in election.ballots().enumerate() {
        trace!("Processing ballot {i} = {:?}", ballot);

        let mut votes_distributed = 0;
        for ranking in ballot.order() {
            let rank_len = ranking.len();
            let weight = if votes_distributed + rank_len <= votes_per_ballot {
                // Ballot still has enough power for all candidates at this rank.
                R::one()
            } else {
                // Split the remaining power equally among candidates at this rank.
                R::from_usize(votes_per_ballot - votes_distributed) / I::from_usize(rank_len)
            };
            trace!(
                "  - {weight} * {:?}",
                ranking.iter().map(|&x| x.into()).collect::<Vec<_>>()
            );

            let ballot_count = R::from_usize(ballot.count());
            for &candidate in ranking {
                sum[candidate.into()] += &weight * &ballot_count;
            }

            votes_distributed += rank_len;
            if votes_distributed >= votes_per_ballot {
                break;
            }
        }
    }

    let mut order: Vec<usize> = (0..election.num_candidates).collect();
    order.sort_by(|&i, &j| {
        sum[j].partial_cmp(&sum[i]).unwrap().then_with(|| {
            election
                .tie_order
                .get(&i)
                .unwrap()
                .cmp(election.tie_order.get(&j).unwrap())
        })
    });

    debug!("Sums:");
    for (i, &c) in order.iter().enumerate() {
        debug!("    [{i}] {} = {}", election.candidates[c].nickname, sum[c]);
    }

    let elected: Vec<usize> = order.iter().cloned().take(election.num_seats).collect();
    let not_elected: Vec<usize> = order
        .iter()
        .cloned()
        .rev()
        .take(election.num_candidates - election.num_seats)
        .filter(|&i| !election.candidates[i].is_withdrawn)
        .collect();

    writeln!(stdout, "Action: Count Complete")?;
    info!("Elected:");
    for (i, &id) in elected.iter().enumerate() {
        info!("    [{}] {}", i, election.candidates[id].nickname);
        writeln!(stdout, "Action: Elect: {}", election.candidates[id].name)?;
    }

    info!("Not elected:");
    for (i, &id) in not_elected.iter().enumerate() {
        info!(
            "    [{}] {}",
            election.num_candidates - i - 1,
            election.candidates[id].nickname
        );
        writeln!(stdout, "Action: Defeat: {}", election.candidates[id].name)?;
    }

    writeln!(stdout, "Action: Count Complete")?;
    for &id in elected.iter() {
        writeln!(
            stdout,
            "\tElected:  {} ({})",
            election.candidates[id].name, sum[id]
        )?;
    }
    for &id in not_elected.iter().rev() {
        writeln!(
            stdout,
            "\tDefeated: {} ({})",
            election.candidates[id].name, sum[id]
        )?;
    }

    let result = ElectionResult {
        elected,
        not_elected,
    };
    Ok(result)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::arithmetic::{FixedDecimal9, Integer64};
    use crate::types::{Ballot, Candidate};

    fn make_election() -> Election {
        Election::builder()
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
                Ballot::new(1, [vec![0], vec![2, 3, 4], vec![6]]),
                Ballot::new(2, [vec![2], vec![3, 4], vec![6, 7]]),
                Ballot::new(3, [vec![2, 3], vec![4, 6], vec![7, 8]]),
                Ballot::new(4, [vec![3, 4], vec![6, 7], vec![8, 0]]),
                Ballot::new(5, [vec![4], vec![6, 7, 8], vec![0]]),
                Ballot::new(6, [vec![6], vec![7, 8, 0], vec![2]]),
                Ballot::new(7, [vec![6, 7], vec![8, 0], vec![2, 3]]),
                Ballot::new(8, [vec![7, 8], vec![0, 2], vec![3, 4]]),
                Ballot::new(9, [vec![8, 0], vec![2, 3], vec![4]]),
            ])
            .build()
    }

    #[test]
    fn test_plurality_block_voting() {
        let election = make_election();

        let mut buf = Vec::new();
        let result = plurality_block_voting::<Integer64, FixedDecimal9>(
            &mut buf,
            &election,
            "package name",
            None,
        )
        .unwrap();
        assert_eq!(
            result,
            ElectionResult {
                elected: vec![8, 0, 7, 2, 4],
                not_elected: vec![3, 6]
            }
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	package name
	Rule: Simulated Plurality Block Voting
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 5
	Ballots: 45
	Votes per ballot: 5

Action: Count Complete
Action: Elect: Jalapeno
Action: Elect: Apple
Action: Elect: Hazelnut
Action: Elect: Cherry
Action: Elect: Eggplant
Action: Defeat: Date
Action: Defeat: Grape
Action: Count Complete
	Elected:  Jalapeno (38.500000000)
	Elected:  Apple (38.000000000)
	Elected:  Hazelnut (33.500000000)
	Elected:  Cherry (32.500000000)
	Elected:  Eggplant (28.000000000)
	Defeated: Grape (28.000000000)
	Defeated: Date (26.500000000)
"
        );
    }

    #[test]
    fn test_plurality_block_voting_1() {
        let election = make_election();

        let mut buf = Vec::new();
        let result = plurality_block_voting::<Integer64, FixedDecimal9>(
            &mut buf,
            &election,
            "package name",
            Some(1),
        )
        .unwrap();
        assert_eq!(
            result,
            ElectionResult {
                elected: vec![6, 8, 7, 4, 0],
                not_elected: vec![3, 2]
            }
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	package name
	Rule: Simulated Plurality Block Voting
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 5
	Ballots: 45
	Votes per ballot: 1

Action: Count Complete
Action: Elect: Grape
Action: Elect: Jalapeno
Action: Elect: Hazelnut
Action: Elect: Eggplant
Action: Elect: Apple
Action: Defeat: Date
Action: Defeat: Cherry
Action: Count Complete
	Elected:  Grape (9.500000000)
	Elected:  Jalapeno (8.500000000)
	Elected:  Hazelnut (7.500000000)
	Elected:  Eggplant (7.000000000)
	Elected:  Apple (5.500000000)
	Defeated: Cherry (3.500000000)
	Defeated: Date (3.500000000)
"
        );
    }

    #[test]
    fn test_plurality_block_voting_2() {
        let election = make_election();

        let mut buf = Vec::new();
        let result = plurality_block_voting::<Integer64, FixedDecimal9>(
            &mut buf,
            &election,
            "package name",
            Some(2),
        )
        .unwrap();
        assert_eq!(
            result,
            ElectionResult {
                elected: vec![8, 7, 6, 0, 4],
                not_elected: vec![2, 3]
            }
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	package name
	Rule: Simulated Plurality Block Voting
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 5
	Ballots: 45
	Votes per ballot: 2

Action: Count Complete
Action: Elect: Jalapeno
Action: Elect: Hazelnut
Action: Elect: Grape
Action: Elect: Apple
Action: Elect: Eggplant
Action: Defeat: Cherry
Action: Defeat: Date
Action: Count Complete
	Elected:  Jalapeno (20.666666663)
	Elected:  Hazelnut (18.666666663)
	Elected:  Grape (14.666666665)
	Elected:  Apple (11.999999998)
	Elected:  Eggplant (10.333333333)
	Defeated: Date (8.333333333)
	Defeated: Cherry (5.333333333)
"
        );
    }

    #[test]
    fn test_plurality_block_voting_3() {
        let election = make_election();

        let mut buf = Vec::new();
        let result = plurality_block_voting::<Integer64, FixedDecimal9>(
            &mut buf,
            &election,
            "package name",
            Some(3),
        )
        .unwrap();
        assert_eq!(
            result,
            ElectionResult {
                elected: vec![8, 7, 0, 6, 2],
                not_elected: vec![4, 3]
            }
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	package name
	Rule: Simulated Plurality Block Voting
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 5
	Ballots: 45
	Votes per ballot: 3

Action: Count Complete
Action: Elect: Jalapeno
Action: Elect: Hazelnut
Action: Elect: Apple
Action: Elect: Grape
Action: Elect: Cherry
Action: Defeat: Eggplant
Action: Defeat: Date
Action: Count Complete
	Elected:  Jalapeno (27.833333326)
	Elected:  Hazelnut (24.333333326)
	Elected:  Apple (21.499999996)
	Elected:  Grape (19.833333330)
	Elected:  Cherry (14.166666666)
	Defeated: Date (14.166666666)
	Defeated: Eggplant (13.166666666)
"
        );
    }

    #[test]
    fn test_plurality_block_voting_4() {
        let election = make_election();

        let mut buf = Vec::new();
        let result = plurality_block_voting::<Integer64, FixedDecimal9>(
            &mut buf,
            &election,
            "package name",
            Some(4),
        )
        .unwrap();
        assert_eq!(
            result,
            ElectionResult {
                elected: vec![8, 0, 7, 6, 2],
                not_elected: vec![4, 3]
            }
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	package name
	Rule: Simulated Plurality Block Voting
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 5
	Ballots: 45
	Votes per ballot: 4

Action: Count Complete
Action: Elect: Jalapeno
Action: Elect: Apple
Action: Elect: Hazelnut
Action: Elect: Grape
Action: Elect: Cherry
Action: Defeat: Eggplant
Action: Defeat: Date
Action: Count Complete
	Elected:  Jalapeno (35.000000000)
	Elected:  Apple (31.000000000)
	Elected:  Hazelnut (31.000000000)
	Elected:  Grape (26.000000000)
	Elected:  Cherry (23.000000000)
	Defeated: Date (19.000000000)
	Defeated: Eggplant (15.000000000)
"
        );
    }

    #[test]
    fn test_plurality_block_voting_6() {
        let election = make_election();

        let mut buf = Vec::new();
        let result = plurality_block_voting::<Integer64, FixedDecimal9>(
            &mut buf,
            &election,
            "package name",
            Some(6),
        )
        .unwrap();
        assert_eq!(
            result,
            ElectionResult {
                elected: vec![8, 0, 2, 7, 3],
                not_elected: vec![6, 4]
            }
        );
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r"
Election: Vegetable contest

	package name
	Rule: Simulated Plurality Block Voting
	Arithmetic: fixed-point decimal arithmetic (9 places)
	Seats: 5
	Ballots: 45
	Votes per ballot: 6

Action: Count Complete
Action: Elect: Jalapeno
Action: Elect: Apple
Action: Elect: Cherry
Action: Elect: Hazelnut
Action: Elect: Date
Action: Defeat: Grape
Action: Defeat: Eggplant
Action: Count Complete
	Elected:  Jalapeno (42.000000000)
	Elected:  Apple (40.000000000)
	Elected:  Cherry (36.000000000)
	Elected:  Hazelnut (35.000000000)
	Elected:  Date (34.000000000)
	Defeated: Eggplant (32.000000000)
	Defeated: Grape (28.000000000)
"
        );
    }
}
