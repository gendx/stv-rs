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
use rand::SeedableRng;
use rand::{thread_rng, RngCore};
use rand_chacha::ChaChaRng;
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

    let file = File::create("rand_hypergeometric_10k.blt")?;
    write_blt_hypergeometric(&mut thread_rng(), &mut BufWriter::new(file), 10000)?;

    let file = File::create("rand_hypergeometric_100k.blt")?;
    write_blt_hypergeometric(
        &mut ChaChaRng::seed_from_u64(42),
        &mut BufWriter::new(file),
        100000,
    )?;

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

    #[test]
    fn test_blt_2x10() {
        let mut buf = Vec::new();
        let mut rng = ChaChaRng::seed_from_u64(42);
        write_blt_2x10(&mut rng, &mut buf, /* ballot_count = */ 7).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
51 jalapeno=banana radish=pear soy=tomato nut=orange hazelnut=apple mushroom=eggplant quinoa=litchi fig=date kiwi=cherry grape=vanilla 0
64 soy=eggplant kiwi=tomato fig=banana nut=radish cherry=apple jalapeno=hazelnut orange=quinoa pear=date grape=litchi mushroom=vanilla 0
92 jalapeno=grape hazelnut=vanilla nut=quinoa tomato=cherry litchi=date orange=fig radish=eggplant apple=soy mushroom=kiwi pear=banana 0
46 quinoa=vanilla kiwi=tomato pear=apple soy=hazelnut fig=nut banana=mushroom cherry=orange date=eggplant grape=jalapeno radish=litchi 0
71 radish=eggplant cherry=orange quinoa=pear banana=litchi date=fig mushroom=tomato vanilla=apple kiwi=jalapeno grape=hazelnut nut=soy 0
52 eggplant=pear vanilla=nut hazelnut=kiwi mushroom=soy cherry=radish banana=orange date=grape apple=litchi quinoa=fig jalapeno=tomato 0
73 orange=mushroom date=hazelnut banana=cherry eggplant=grape kiwi=fig quinoa=tomato soy=radish pear=litchi vanilla=jalapeno nut=apple 0
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
        let mut rng = ChaChaRng::seed_from_u64(42);
        write_blt_5x4(&mut rng, &mut buf, /* ballot_count = */ 7).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
51 jalapeno=banana=radish=pear=soy tomato=nut=orange=hazelnut=apple mushroom=eggplant=quinoa=litchi=fig date=kiwi=cherry=grape=vanilla 0
64 soy=eggplant=kiwi=tomato=fig banana=nut=radish=cherry=apple jalapeno=hazelnut=orange=quinoa=pear date=grape=litchi=mushroom=vanilla 0
92 jalapeno=grape=hazelnut=vanilla=nut quinoa=tomato=cherry=litchi=date orange=fig=radish=eggplant=apple soy=mushroom=kiwi=pear=banana 0
46 quinoa=vanilla=kiwi=tomato=pear apple=soy=hazelnut=fig=nut banana=mushroom=cherry=orange=date eggplant=grape=jalapeno=radish=litchi 0
71 radish=eggplant=cherry=orange=quinoa pear=banana=litchi=date=fig mushroom=tomato=vanilla=apple=kiwi jalapeno=grape=hazelnut=nut=soy 0
52 eggplant=pear=vanilla=nut=hazelnut kiwi=mushroom=soy=cherry=radish banana=orange=date=grape=apple litchi=quinoa=fig=jalapeno=tomato 0
73 orange=mushroom=date=hazelnut=banana cherry=eggplant=grape=kiwi=fig quinoa=tomato=soy=radish=pear litchi=vanilla=jalapeno=nut=apple 0
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
        let mut rng = ChaChaRng::seed_from_u64(42);
        write_blt_geometric(&mut rng, &mut buf, /* ballot_count = */ 7).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
51 cherry=date=hazelnut=jalapeno=mushroom=orange=pear=quinoa=radish=vanilla kiwi=litchi=tomato eggplant=nut=soy grape banana fig apple 0
28 apple=cherry=grape=hazelnut=jalapeno=litchi=mushroom=nut=pear=soy=tomato=vanilla kiwi=orange=quinoa=radish banana=fig date eggplant 0
47 cherry=date=hazelnut=litchi=nut=pear=quinoa=radish=tomato=vanilla grape=orange banana=fig=jalapeno=kiwi=mushroom apple=soy eggplant 0
65 fig=hazelnut=jalapeno=litchi=mushroom=nut=orange=pear=radish=soy=tomato=vanilla apple=kiwi banana=cherry=quinoa eggplant date=grape 0
9 banana=cherry=eggplant=hazelnut=jalapeno=pear=quinoa=radish=soy=tomato=vanilla litchi=mushroom=nut=orange apple=fig=grape date kiwi 0
47 fig=grape=mushroom=orange=pear=quinoa=soy=tomato=vanilla cherry=hazelnut=jalapeno=radish eggplant=nut litchi apple=kiwi banana=date 0
65 banana=cherry=hazelnut=jalapeno=kiwi=litchi=mushroom=orange=quinoa=soy=vanilla fig=grape=radish=tomato eggplant=nut=pear date apple 0
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
        let mut rng = ChaChaRng::seed_from_u64(42);
        write_blt_hypergeometric(&mut rng, &mut buf, /* ballot_count = */ 7).unwrap();

        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            r#"20 10
[nick apple banana cherry date eggplant fig grape hazelnut jalapeno kiwi litchi mushroom nut orange pear quinoa radish soy tomato vanilla]
51 banana apple=cherry=grape eggplant date fig=jalapeno=mushroom=soy kiwi hazelnut=quinoa litchi orange=pear nut tomato vanilla radish 0
46 banana apple date cherry=fig eggplant=mushroom=nut grape hazelnut=jalapeno=kiwi=orange=quinoa radish=soy pear=vanilla litchi tomato 0
33 cherry jalapeno banana=date apple=grape eggplant nut fig=litchi=pear hazelnut=mushroom=quinoa kiwi=tomato orange vanilla soy radish 0
16 banana date apple=eggplant=fig=jalapeno grape=kiwi pear litchi=mushroom cherry hazelnut=nut vanilla orange=quinoa radish soy tomato 0
2 banana cherry=eggplant apple=hazelnut=jalapeno date nut=radish grape=orange fig=pear kiwi=litchi=tomato quinoa=soy vanilla mushroom 0
20 apple banana=litchi cherry=nut fig=grape=hazelnut=kiwi date=eggplant jalapeno radish pear=tomato mushroom quinoa orange=soy vanilla 0
44 banana apple cherry=grape date=nut eggplant=fig hazelnut=jalapeno pear kiwi=orange radish litchi=mushroom vanilla tomato quinoa=soy 0
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
