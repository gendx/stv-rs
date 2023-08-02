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

//! Script to generate random ballot files.

#![forbid(missing_docs, unsafe_code)]

use rand::distributions::{Distribution, Uniform};
use rand::seq::index::sample;
use rand::{thread_rng, RngCore};
use rand_distr::{Geometric, Hypergeometric};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Result, Write};
use stv_rs::blt::{write_blt, CandidateFormat, WriteTieOrder};
use stv_rs::types::{Ballot, Candidate, Election, ElectionBuilder};

static VEGETABLES: [&str; 20] = [
    "apple", "banana", "cherry", "date", "eggplant", "fig", "grape", "hazelnut", "jalapeno",
    "kiwi", "litchi", "mushroom", "nut", "orange", "pear", "quinoa", "radish", "soy", "tomato",
    "vanilla",
];

fn main() -> Result<()> {
    let file = File::create("rand_2x10.blt")?;
    write_blt_2x10(&mut thread_rng(), &mut BufWriter::new(file), 1000)?;

    let file = File::create("rand_5x4.blt")?;
    write_blt_5x4(&mut thread_rng(), &mut BufWriter::new(file), 1000)?;

    let file = File::create("rand_geometric.blt")?;
    write_blt_geometric(&mut thread_rng(), &mut BufWriter::new(file), 1000)?;

    let file = File::create("rand_hypergeometric.blt")?;
    write_blt_hypergeometric(&mut thread_rng(), &mut BufWriter::new(file), 1000)?;

    Ok(())
}

fn new_election_builder() -> ElectionBuilder {
    Election::builder()
        .title("Vegetable contest")
        .num_seats(10)
        .candidates(
            VEGETABLES
                .iter()
                .map(|&nickname| Candidate::new(nickname, /* is_withdrawn = */ false))
                .collect::<Vec<_>>(),
        )
}

fn write_blt_2x10(
    rng: &mut impl RngCore,
    output: &mut impl Write,
    ballot_count: usize,
) -> Result<()> {
    let election = new_election_builder()
        .ballots(generate_tuples(rng, ballot_count, 2))
        .build();
    write_blt(
        output,
        &election,
        WriteTieOrder::Never,
        CandidateFormat::Nicknames,
    )?;
    Ok(())
}

fn write_blt_5x4(
    rng: &mut impl RngCore,
    output: &mut impl Write,
    ballot_count: usize,
) -> Result<()> {
    let election = new_election_builder()
        .ballots(generate_tuples(rng, ballot_count, 5))
        .build();
    write_blt(
        output,
        &election,
        WriteTieOrder::Never,
        CandidateFormat::Nicknames,
    )?;
    Ok(())
}

fn write_blt_geometric(
    rng: &mut impl RngCore,
    output: &mut impl Write,
    ballot_count: usize,
) -> Result<()> {
    let election = new_election_builder()
        .ballots(generate_geometric(rng, ballot_count))
        .build();
    write_blt(
        output,
        &election,
        WriteTieOrder::Never,
        CandidateFormat::Nicknames,
    )?;
    Ok(())
}

fn write_blt_hypergeometric(
    rng: &mut impl RngCore,
    output: &mut impl Write,
    ballot_count: usize,
) -> Result<()> {
    let election = new_election_builder()
        .ballots(generate_hypergeometric(rng, ballot_count))
        .build();
    write_blt(
        output,
        &election,
        WriteTieOrder::Never,
        CandidateFormat::Nicknames,
    )?;
    Ok(())
}

fn generate_geometric(rng: &mut impl RngCore, ballot_count: usize) -> Vec<Ballot> {
    let distributions = (0..20)
        .map(|i| Geometric::new(0.3 + i as f64 / 50.0).unwrap())
        .collect::<Vec<_>>();
    generate_distributions(rng, ballot_count, &distributions)
}

fn generate_hypergeometric(rng: &mut impl RngCore, ballot_count: usize) -> Vec<Ballot> {
    let distributions = (0..20)
        .map(|i| Hypergeometric::new(100, 50, 20 + i).unwrap())
        .collect::<Vec<_>>();
    generate_distributions(rng, ballot_count, &distributions)
}

fn generate_tuples(rng: &mut impl RngCore, ballot_count: usize, tuple_size: usize) -> Vec<Ballot> {
    let count_dist = Uniform::from(1..100);

    let mut ballots = Vec::new();
    for _ in 0..ballot_count {
        let count = count_dist.sample(rng);
        let order = sample(rng, 20, 20);
        ballots.push(Ballot::new(
            count,
            order
                .into_vec()
                .chunks(tuple_size)
                .map(|rank| rank.to_vec())
                .collect::<Vec<_>>(),
        ));
    }
    ballots
}

fn generate_distributions<D: Distribution<u64>>(
    rng: &mut impl RngCore,
    ballot_count: usize,
    distributions: &[D],
) -> Vec<Ballot> {
    let count_dist = Uniform::from(1..100);

    let mut ballots = Vec::new();
    for _ in 0..ballot_count {
        let count = count_dist.sample(rng);

        let mut order: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (i, d) in distributions.iter().enumerate() {
            let value = d.sample(rng);
            order.entry(value).or_default().push(i);
        }

        ballots.push(Ballot::new(count, order.into_values().collect::<Vec<_>>()));
    }

    ballots
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_blt_2x10() {
        let mut buf = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);
        write_blt_2x10(&mut rng, &mut buf, /* ballot_count = */ 7).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
53 soy=orange kiwi=vanilla pear=nut grape=apple date=quinoa cherry=eggplant banana=tomato hazelnut=jalapeno radish=mushroom litchi=fig 0
66 fig=soy litchi=mushroom nut=date grape=jalapeno eggplant=hazelnut pear=kiwi tomato=banana quinoa=orange vanilla=apple cherry=radish 0
64 radish=fig jalapeno=kiwi litchi=grape hazelnut=nut cherry=quinoa pear=orange banana=vanilla date=eggplant mushroom=tomato apple=soy 0
38 eggplant=vanilla grape=tomato pear=kiwi nut=banana hazelnut=jalapeno soy=quinoa orange=radish apple=cherry litchi=date fig=mushroom 0
72 kiwi=vanilla orange=radish jalapeno=cherry eggplant=fig hazelnut=litchi banana=tomato nut=pear mushroom=date soy=grape apple=quinoa 0
61 banana=eggplant kiwi=quinoa date=vanilla fig=litchi cherry=hazelnut tomato=orange soy=jalapeno mushroom=radish grape=apple nut=pear 0
29 tomato=apple quinoa=pear jalapeno=hazelnut orange=nut litchi=grape date=radish cherry=vanilla fig=mushroom kiwi=eggplant soy=banana 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Fig"
"Grape"
"Hazelnut"
"Jalapeno"
"Kiwi"
"Litchi"
"Mushroom"
"Nut"
"Orange"
"Pear"
"Quinoa"
"Radish"
"Soy"
"Tomato"
"Vanilla"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_blt_5x4() {
        let mut buf = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);
        write_blt_5x4(&mut rng, &mut buf, /* ballot_count = */ 7).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
53 soy=orange=kiwi=vanilla=pear nut=grape=apple=date=quinoa cherry=eggplant=banana=tomato=hazelnut jalapeno=radish=mushroom=litchi=fig 0
66 fig=soy=litchi=mushroom=nut date=grape=jalapeno=eggplant=hazelnut pear=kiwi=tomato=banana=quinoa orange=vanilla=apple=cherry=radish 0
64 radish=fig=jalapeno=kiwi=litchi grape=hazelnut=nut=cherry=quinoa pear=orange=banana=vanilla=date eggplant=mushroom=tomato=apple=soy 0
38 eggplant=vanilla=grape=tomato=pear kiwi=nut=banana=hazelnut=jalapeno soy=quinoa=orange=radish=apple cherry=litchi=date=fig=mushroom 0
72 kiwi=vanilla=orange=radish=jalapeno cherry=eggplant=fig=hazelnut=litchi banana=tomato=nut=pear=mushroom date=soy=grape=apple=quinoa 0
61 banana=eggplant=kiwi=quinoa=date vanilla=fig=litchi=cherry=hazelnut tomato=orange=soy=jalapeno=mushroom radish=grape=apple=nut=pear 0
29 tomato=apple=quinoa=pear=jalapeno hazelnut=orange=nut=litchi=grape date=radish=cherry=vanilla=fig mushroom=kiwi=eggplant=soy=banana 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Fig"
"Grape"
"Hazelnut"
"Jalapeno"
"Kiwi"
"Litchi"
"Mushroom"
"Nut"
"Orange"
"Pear"
"Quinoa"
"Radish"
"Soy"
"Tomato"
"Vanilla"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_blt_geometric() {
        let mut buf = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);
        write_blt_geometric(&mut rng, &mut buf, /* ballot_count = */ 7).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
53 eggplant=fig=nut=orange=pear=radish=tomato apple=jalapeno=kiwi=litchi=quinoa=vanilla date=grape=mushroom=soy cherry=hazelnut banana 0
49 date=jalapeno=orange=radish=soy=vanilla apple=eggplant=fig=kiwi=litchi=mushroom=nut=pear=quinoa=tomato banana=hazelnut cherry=grape 0
1 apple=cherry=date=grape=jalapeno=kiwi=nut=quinoa=soy=tomato=vanilla fig=litchi=orange eggplant=pear mushroom banana=radish hazelnut 0
35 apple=cherry=fig=hazelnut=nut=orange=pear=quinoa=radish=soy=vanilla banana=grape=jalapeno kiwi=litchi mushroom=tomato eggplant date 0
20 banana=jalapeno=nut=orange=quinoa=radish=soy=tomato=vanilla cherry=eggplant=grape=litchi=mushroom kiwi=pear date hazelnut fig apple 0
61 kiwi=litchi=pear=radish=soy banana=cherry=eggplant=fig=jalapeno=mushroom=orange=vanilla grape=hazelnut=nut date apple=tomato quinoa 0
52 date=fig=jalapeno=mushroom=nut=orange=pear=quinoa=radish=soy=vanilla apple=grape=kiwi=litchi=tomato cherry=hazelnut eggplant banana 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Fig"
"Grape"
"Hazelnut"
"Jalapeno"
"Kiwi"
"Litchi"
"Mushroom"
"Nut"
"Orange"
"Pear"
"Quinoa"
"Radish"
"Soy"
"Tomato"
"Vanilla"
"Vegetable contest"
"#
        );
    }

    #[test]
    fn test_blt_hypergeometric() {
        let mut buf = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);
        write_blt_hypergeometric(&mut rng, &mut buf, /* ballot_count = */ 7).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
53 fig apple=banana=cherry=grape eggplant=kiwi jalapeno=mushroom date=hazelnut=litchi=orange=soy pear nut=quinoa=tomato radish vanilla 0
55 cherry banana=grape apple hazelnut fig date kiwi=litchi=nut=quinoa=vanilla jalapeno=mushroom=orange=tomato eggplant pear=radish soy 0
27 apple banana=jalapeno date=litchi=mushroom=nut cherry=hazelnut=pear kiwi=tomato eggplant=fig=quinoa=soy grape orange=radish vanilla 0
10 eggplant banana cherry=grape apple=kiwi litchi date=fig hazelnut jalapeno=mushroom=soy=vanilla orange pear nut quinoa radish=tomato 0
58 date banana jalapeno cherry=hazelnut apple=eggplant=grape=mushroom fig kiwi=soy quinoa litchi=nut orange=pear=vanilla tomato radish 0
5 apple jalapeno cherry=date=litchi fig eggplant=mushroom=nut banana=grape=kiwi=tomato hazelnut=orange pear=radish soy=vanilla quinoa 0
27 date cherry apple banana=eggplant=fig grape jalapeno=mushroom=nut litchi orange=pear radish hazelnut=soy kiwi=quinoa tomato vanilla 0
0
"Apple"
"Banana"
"Cherry"
"Date"
"Eggplant"
"Fig"
"Grape"
"Hazelnut"
"Jalapeno"
"Kiwi"
"Litchi"
"Mushroom"
"Nut"
"Orange"
"Pear"
"Quinoa"
"Radish"
"Soy"
"Tomato"
"Vanilla"
"Vegetable contest"
"#
        );
    }
}
