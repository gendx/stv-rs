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

//! Module to parse STV ballot files.

use crate::types::{Ballot, Candidate, Election};
use log::{info, trace, warn};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

/// Options to control the parsing process.
pub struct ParsingOptions {
    /// Whether to remove withdrawn candidates from the ballots they appear in.
    pub remove_withdrawn_candidates: bool,
    /// Whether to remove ballots that rank no candidate. When
    /// `remove_withdrawn_candidates` is true, this also removes ballots
    /// that only rank withdrawn candidates.
    pub remove_empty_ballots: bool,
}

// TODO: Remove unwrap()s.
/// Parses a ballot file into an election input.
pub fn parse_election(
    input: impl BufRead,
    options: ParsingOptions,
) -> Result<Election, Box<dyn std::error::Error>> {
    let re_count = Regex::new(r"^([0-9]+) ([0-9]+)$").unwrap();
    let re_option = Regex::new(r"^\[[a-z]+(?: [a-z][a-z0-9]*)+\]$").unwrap();
    let re_ballot = Regex::new(r"^([0-9]+)((?: [a-z0-9=]*)*) 0$").unwrap();

    let mut lines = input.lines().peekable();

    let header = lines.next().unwrap().unwrap();
    let cap_count = re_count.captures(&header).unwrap();
    let num_candidates = cap_count.get(1).unwrap().as_str().parse::<usize>().unwrap();
    let num_seats = cap_count.get(2).unwrap().as_str().parse::<usize>().unwrap();

    info!("{num_seats} seats / {num_candidates} candidates");

    // Parse the options
    let mut nicknames = None;
    let mut withdrawn: HashSet<String> = HashSet::new();
    let mut tie = None;
    while let Some(line) = lines.peek() {
        let line = line.as_ref().unwrap();
        if !re_option.is_match(line) {
            break;
        }

        let mut items = line[1..line.len() - 1].split(' ');
        let title = items.next().unwrap();

        match title {
            "nick" => {
                let values = items.map(|x| x.to_owned()).collect::<Vec<String>>();
                info!("Nicknames: {values:?}");
                nicknames = Some(values);
            }
            "withdrawn" => {
                let values = items.map(|x| x.to_owned()).collect::<Vec<String>>();
                info!("Withdrawn: {values:?}");
                withdrawn = values.into_iter().collect::<HashSet<String>>();
            }
            "tie" => {
                let values = items.map(|x| x.to_owned()).collect::<Vec<String>>();
                info!("Tie-break order: {values:?}");
                tie = Some(values);
            }
            _ => warn!("Unknown option: {title}"),
        }

        lines.next();
    }

    let nicknames: Vec<String> = nicknames.unwrap();
    info!("Candidates (by nickname): {nicknames:?}");
    assert_eq!(nicknames.len(), num_candidates);

    let hash_nicknames: HashMap<&str, usize> = nicknames
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();

    let tie_order: HashMap<usize, usize> = {
        match tie {
            None => (0..num_candidates).map(|i| (i, i)).collect(),
            Some(tie) => {
                assert_eq!(
                    tie.len(),
                    num_candidates,
                    "Tie-break order must mention all candidates"
                );
                let mut tie_order = HashMap::new();
                for (i, c) in tie.iter().enumerate() {
                    let id = *hash_nicknames.get(c.as_str()).unwrap();
                    assert!(
                        tie_order.insert(id, i).is_none(),
                        "Candidate mentioned twice in tie order: {c}",
                    );
                }
                tie_order
            }
        }
    };

    let mut ballots = Vec::new();
    loop {
        let line = lines.next().unwrap().unwrap();
        if line == "0" {
            break;
        }
        match re_ballot.captures(&line) {
            Some(cap_ballots) => {
                let count = cap_ballots
                    .get(1)
                    .unwrap()
                    .as_str()
                    .parse::<usize>()
                    .unwrap();
                let order_str = cap_ballots.get(2).unwrap().as_str();
                let order = order_str.split(' ').filter_map(|level| {
                    if level.is_empty() {
                        None
                    } else {
                        let mut level_candidates = level
                            .split('=')
                            .filter_map(|candidate| {
                                if options.remove_withdrawn_candidates
                                    && withdrawn.contains(candidate)
                                {
                                    None
                                } else {
                                    Some(*hash_nicknames.get(candidate).unwrap())
                                }
                            })
                            .peekable();
                        if level_candidates.peek().is_none() {
                            None
                        } else {
                            Some(level_candidates)
                        }
                    }
                });

                let ballot = Ballot::new(count, order);
                trace!(
                    "Parsed ballot: count {count} for {:?}",
                    ballot.order().collect::<Vec<_>>()
                );
                if options.remove_empty_ballots && ballot.order_len() == 0 {
                    warn!("Removing ballot that is empty or contains only withdrawn candidates: {line}");
                } else {
                    ballot.validate();
                    ballots.push(ballot);
                }
            }
            None => {
                warn!("Ignored line: {line:?}");
            }
        }
    }

    let num_ballots = ballots.iter().map(|b| b.count()).sum::<usize>();
    info!("Number of ballots: {num_ballots}");

    let candidates: Vec<Candidate> = nicknames
        .into_iter()
        .map(|nickname| {
            let is_withdrawn = withdrawn.contains(&nickname);
            Candidate {
                name: remove_quotes(&lines.next().unwrap().unwrap()).to_string(),
                nickname,
                is_withdrawn,
            }
        })
        .collect();

    let title = remove_quotes(&lines.next().unwrap().unwrap()).to_string();
    info!("Election title: {title}");

    Ok(Election {
        title,
        num_candidates,
        num_seats,
        num_ballots,
        candidates,
        ballots,
        tie_order,
    })
}

/// Removes the leading and trailing quotes. The input string must start with a
/// double-quote character and end with a double-quote character -- only these
/// two characters are removed.
fn remove_quotes(x: &str) -> &str {
    // TODO: Implement a more robust parsing of quoted strings.
    assert!(x.len() >= 2);
    assert_eq!(*x.as_bytes().first().unwrap(), b'"');
    assert_eq!(*x.as_bytes().last().unwrap(), b'"');
    &x[1..x.len() - 1]
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::log_tester::ThreadLocalLogger;
    use log::Level::{Info, Warn};
    use std::io::Cursor;

    #[test]
    fn test_remove_quotes() {
        assert_eq!(remove_quotes("\"foo\""), "foo");
        assert_eq!(remove_quotes("\"Hello world\""), "Hello world");
    }

    #[test]
    #[should_panic(expected = "assertion failed: x.len() >= 2")]
    fn test_remove_quotes_empty() {
        remove_quotes("");
    }

    #[test]
    #[should_panic(expected = "assertion failed: x.len() >= 2")]
    fn test_remove_quotes_short() {
        remove_quotes("\"");
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed\n  left: 102\n right: 34")]
    fn test_remove_quotes_no_quotes() {
        remove_quotes("foo");
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed\n  left: 225\n right: 34")]
    fn test_remove_quotes_mid_utf8() {
        remove_quotes("\u{1234}foo\"");
    }

    /// Returns baseline parsing options for tests where they don't matter.
    fn basic_parsing_options() -> ParsingOptions {
        ParsingOptions {
            remove_withdrawn_candidates: true,
            remove_empty_ballots: true,
        }
    }

    #[test]
    fn test_parse_election() {
        let file = r#"5 2
[nick apple banana cherry date eggplant]
[tie cherry apple eggplant banana date]
3 apple cherry eggplant date banana 0
3 date=eggplant banana=cherry=apple 0
42 cherry 0
123 banana date 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Vegetable contest"
"#;
        let logger = ThreadLocalLogger::start();
        let election = parse_election(Cursor::new(file), basic_parsing_options()).unwrap();

        assert_eq!(
            election,
            Election::builder()
                .title("Vegetable contest")
                .num_seats(2)
                .candidates([
                    Candidate::new("apple", false),
                    Candidate::new("banana", false),
                    Candidate::new("cherry", false),
                    Candidate::new("date", false),
                    Candidate::new("eggplant", false),
                ])
                .ballots(vec![
                    Ballot::new(3, [vec![0], vec![2], vec![4], vec![3], vec![1]]),
                    Ballot::new(3, [vec![3, 4], vec![1, 2, 0]]),
                    Ballot::new(42, [vec![2]]),
                    Ballot::new(123, [vec![1], vec![3]]),
                ])
                .check_num_ballots(171)
                .tie_order([2, 0, 4, 1, 3])
                .build()
        );
        logger.check_target_logs(
            "stv_rs::parse",
            [
                (Info, "2 seats / 5 candidates"),
                (Info, "Nicknames: [\"apple\", \"banana\", \"cherry\", \"date\", \"eggplant\"]"),
                (Info, "Tie-break order: [\"cherry\", \"apple\", \"eggplant\", \"banana\", \"date\"]"),
                (Info, "Candidates (by nickname): [\"apple\", \"banana\", \"cherry\", \"date\", \"eggplant\"]"),
                (Info, "Number of ballots: 171"),
                (Info, "Election title: Vegetable contest"),
            ],
        );
    }

    #[test]
    fn test_parse_names_with_digits() {
        let file = r#"3 2
[nick apple ba2nana34 cherry]
[tie cherry apple ba2nana34]
1 apple ba2nana34 0
0
"Apple"
"Ba 2 nana 34"
"Cherry"
"Vegetable contest"
"#;
        let logger = ThreadLocalLogger::start();
        let election = parse_election(Cursor::new(file), basic_parsing_options()).unwrap();

        assert_eq!(
            election,
            Election::builder()
                .title("Vegetable contest")
                .num_seats(2)
                .candidates([
                    Candidate::new("apple", false),
                    Candidate {
                        nickname: "ba2nana34".to_owned(),
                        name: "Ba 2 nana 34".to_owned(),
                        is_withdrawn: false,
                    },
                    Candidate::new("cherry", false),
                ])
                .ballots(vec![Ballot::new(1, [vec![0], vec![1]])])
                .check_num_ballots(1)
                .tie_order([2, 0, 1])
                .build()
        );
        logger.check_target_logs(
            "stv_rs::parse",
            [
                (Info, "2 seats / 3 candidates"),
                (Info, "Nicknames: [\"apple\", \"ba2nana34\", \"cherry\"]"),
                (
                    Info,
                    "Tie-break order: [\"cherry\", \"apple\", \"ba2nana34\"]",
                ),
                (
                    Info,
                    "Candidates (by nickname): [\"apple\", \"ba2nana34\", \"cherry\"]",
                ),
                (Info, "Number of ballots: 1"),
                (Info, "Election title: Vegetable contest"),
            ],
        );
    }

    #[test]
    fn test_parse_withdrawn_keep_all() {
        let file = r#"5 2
[nick apple banana cherry date eggplant]
[withdrawn cherry eggplant]
3 apple cherry eggplant date banana 0
3 date=eggplant banana=cherry=apple 0
42 cherry 0
123 banana date 0
17 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Vegetable contest"
"#;
        let logger = ThreadLocalLogger::start();
        let election = parse_election(
            Cursor::new(file),
            ParsingOptions {
                remove_withdrawn_candidates: false,
                remove_empty_ballots: false,
            },
        )
        .unwrap();

        assert_eq!(
            election,
            Election::builder()
                .title("Vegetable contest")
                .num_seats(2)
                .candidates([
                    Candidate::new("apple", false),
                    Candidate::new("banana", false),
                    Candidate::new("cherry", true),
                    Candidate::new("date", false),
                    Candidate::new("eggplant", true),
                ])
                .ballots(vec![
                    Ballot::new(3, [vec![0], vec![2], vec![4], vec![3], vec![1]]),
                    Ballot::new(3, [vec![3, 4], vec![1, 2, 0]]),
                    Ballot::new(42, [vec![2]]),
                    Ballot::new(123, [vec![1], vec![3]]),
                    Ballot::empties(17),
                ])
                .check_num_ballots(188)
                .build()
        );
        logger.check_target_logs(
            "stv_rs::parse",
            [
                (Info, "2 seats / 5 candidates"),
                (Info, "Nicknames: [\"apple\", \"banana\", \"cherry\", \"date\", \"eggplant\"]"),
                (Info, "Withdrawn: [\"cherry\", \"eggplant\"]"),
                (Info, "Candidates (by nickname): [\"apple\", \"banana\", \"cherry\", \"date\", \"eggplant\"]"),
                (Info, "Number of ballots: 188"),
                (Info, "Election title: Vegetable contest"),
            ],
        );
    }

    #[test]
    fn test_parse_withdrawn_remove_withdrawn() {
        let file = r#"5 2
[nick apple banana cherry date eggplant]
[withdrawn cherry eggplant]
3 apple cherry eggplant date banana 0
3 date=eggplant banana=cherry=apple 0
42 cherry 0
123 banana date 0
17 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Vegetable contest"
"#;
        let logger = ThreadLocalLogger::start();
        let election = parse_election(
            Cursor::new(file),
            ParsingOptions {
                remove_withdrawn_candidates: true,
                remove_empty_ballots: false,
            },
        )
        .unwrap();

        assert_eq!(
            election,
            Election::builder()
                .title("Vegetable contest")
                .num_seats(2)
                .candidates([
                    Candidate::new("apple", false),
                    Candidate::new("banana", false),
                    Candidate::new("cherry", true),
                    Candidate::new("date", false),
                    Candidate::new("eggplant", true),
                ])
                .ballots(vec![
                    Ballot::new(3, [vec![0], vec![3], vec![1]]),
                    Ballot::new(3, [vec![3], vec![1, 0]]),
                    Ballot::empties(42),
                    Ballot::new(123, [vec![1], vec![3]]),
                    Ballot::empties(17),
                ])
                .check_num_ballots(188)
                .build()
        );
        logger.check_target_logs(
            "stv_rs::parse",
            [
                (Info, "2 seats / 5 candidates"),
                (Info, "Nicknames: [\"apple\", \"banana\", \"cherry\", \"date\", \"eggplant\"]"),
                (Info, "Withdrawn: [\"cherry\", \"eggplant\"]"),
                (Info, "Candidates (by nickname): [\"apple\", \"banana\", \"cherry\", \"date\", \"eggplant\"]"),
                (Info, "Number of ballots: 188"),
                (Info, "Election title: Vegetable contest"),
            ],
        );
    }

    #[test]
    fn test_parse_withdrawn_remove_empty_ballots() {
        let file = r#"5 2
[nick apple banana cherry date eggplant]
[withdrawn cherry eggplant]
3 apple cherry eggplant date banana 0
3 date=eggplant banana=cherry=apple 0
42 cherry 0
123 banana date 0
17 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Vegetable contest"
"#;
        let logger = ThreadLocalLogger::start();
        let election = parse_election(
            Cursor::new(file),
            ParsingOptions {
                remove_withdrawn_candidates: false,
                remove_empty_ballots: true,
            },
        )
        .unwrap();

        assert_eq!(
            election,
            Election::builder()
                .title("Vegetable contest")
                .num_seats(2)
                .candidates([
                    Candidate::new("apple", false),
                    Candidate::new("banana", false),
                    Candidate::new("cherry", true),
                    Candidate::new("date", false),
                    Candidate::new("eggplant", true),
                ])
                .ballots(vec![
                    Ballot::new(3, [vec![0], vec![2], vec![4], vec![3], vec![1]]),
                    Ballot::new(3, [vec![3, 4], vec![1, 2, 0]]),
                    Ballot::new(42, [vec![2]]),
                    Ballot::new(123, [vec![1], vec![3]]),
                ])
                .check_num_ballots(171)
                .build()
        );
        logger.check_target_logs(
            "stv_rs::parse",
            [
                (Info, "2 seats / 5 candidates"),
                (Info, "Nicknames: [\"apple\", \"banana\", \"cherry\", \"date\", \"eggplant\"]"),
                (Info, "Withdrawn: [\"cherry\", \"eggplant\"]"),
                (Info, "Candidates (by nickname): [\"apple\", \"banana\", \"cherry\", \"date\", \"eggplant\"]"),
                (Warn, "Removing ballot that is empty or contains only withdrawn candidates: 17 0"),
                (Info, "Number of ballots: 171"),
                (Info, "Election title: Vegetable contest"),
            ],
        );
    }

    #[test]
    fn test_parse_withdrawn_remove_all() {
        let file = r#"5 2
[nick apple banana cherry date eggplant]
[withdrawn cherry eggplant]
3 apple cherry eggplant date banana 0
3 date=eggplant banana=cherry=apple 0
42 cherry 0
123 banana date 0
17 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Vegetable contest"
"#;
        let logger = ThreadLocalLogger::start();
        let election = parse_election(
            Cursor::new(file),
            ParsingOptions {
                remove_withdrawn_candidates: true,
                remove_empty_ballots: true,
            },
        )
        .unwrap();

        assert_eq!(
            election,
            Election::builder()
                .title("Vegetable contest")
                .num_seats(2)
                .candidates([
                    Candidate::new("apple", false),
                    Candidate::new("banana", false),
                    Candidate::new("cherry", true),
                    Candidate::new("date", false),
                    Candidate::new("eggplant", true),
                ])
                .ballots(vec![
                    Ballot::new(3, [vec![0], vec![3], vec![1]]),
                    Ballot::new(3, [vec![3], vec![1, 0]]),
                    Ballot::new(123, [vec![1], vec![3]]),
                ])
                .check_num_ballots(129)
                .build()
        );
        logger.check_target_logs(
            "stv_rs::parse",
            [
                (Info, "2 seats / 5 candidates"),
                (Info, "Nicknames: [\"apple\", \"banana\", \"cherry\", \"date\", \"eggplant\"]"),
                (Info, "Withdrawn: [\"cherry\", \"eggplant\"]"),
                (Info, "Candidates (by nickname): [\"apple\", \"banana\", \"cherry\", \"date\", \"eggplant\"]"),
                (Warn, "Removing ballot that is empty or contains only withdrawn candidates: 42 cherry 0"),
                (Warn, "Removing ballot that is empty or contains only withdrawn candidates: 17 0"),
                (Info, "Number of ballots: 129"),
                (Info, "Election title: Vegetable contest"),
            ],
        );
    }

    #[test]
    fn test_parse_unknown_option() {
        let file = r#"2 1
[nick apple banana]
[unknown foo bar]
1 apple 0
0
"Apple"
"Banana"
"Vegetable contest"
"#;
        let logger = ThreadLocalLogger::start();
        let election = parse_election(Cursor::new(file), basic_parsing_options()).unwrap();

        assert_eq!(
            election,
            Election::builder()
                .title("Vegetable contest")
                .num_seats(1)
                .candidates([
                    Candidate::new("apple", false),
                    Candidate::new("banana", false),
                ])
                .ballots(vec![Ballot::new(1, [vec![0]])])
                .check_num_ballots(1)
                .build()
        );
        logger.check_target_logs(
            "stv_rs::parse",
            [
                (Info, "1 seats / 2 candidates"),
                (Info, "Nicknames: [\"apple\", \"banana\"]"),
                (Warn, "Unknown option: unknown"),
                (Info, "Candidates (by nickname): [\"apple\", \"banana\"]"),
                (Info, "Number of ballots: 1"),
                (Info, "Election title: Vegetable contest"),
            ],
        );
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed: Tie-break order must mention all candidates\n  left: 1\n right: 2"
    )]
    fn test_parse_tie_not_all_candidates() {
        let file = r#"2 1
[nick apple banana]
[tie banana]
1 apple 0
0
"Apple"
"Banana"
"Vegetable contest"
"#;
        let _ = parse_election(Cursor::new(file), basic_parsing_options());
    }

    #[test]
    #[should_panic(expected = "Candidate mentioned twice in tie order: banana")]
    fn test_parse_tie_repeated_candidate() {
        let file = r#"2 1
[nick apple banana]
[tie banana banana]
1 apple 0
0
"Apple"
"Banana"
"Vegetable contest"
"#;
        let _ = parse_election(Cursor::new(file), basic_parsing_options());
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed\n  left: 2\n right: 1")]
    fn test_parse_ballot_repeated_candidate() {
        let file = r#"2 1
[nick apple banana]
1 apple apple 0
0
"Apple"
"Banana"
"Vegetable contest"
"#;
        let _ = parse_election(Cursor::new(file), basic_parsing_options());
    }

    #[test]
    #[should_panic(expected = "called `Option::unwrap()` on a `None` value")]
    fn test_parse_ballot_unknown_nickname() {
        let file = r#"2 1
[nick apple banana]
1 appppppple 0
0
"appppppple"
"bananaaaaa"
"Vegetable contest"
"#;
        let _ = parse_election(Cursor::new(file), basic_parsing_options());
    }
}
