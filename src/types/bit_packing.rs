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

//! Bit-packed representation of ballots.

use crate::types::{count_vec_allocations, Ballot, BallotCursor, BallotView};
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
//use smallvec::SmallVec;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BitPackedBallots {
    /// Bit per candidate.
    bits_per_candidate: u32,
    candidate_mask: u64,
    /// Position of ballots in the below array.
    indices: Vec<usize>,
    /// Bit-packed ballots.
    ballots: Vec<u8>,
}

impl BitPackedBallots {
    pub fn new(num_candidates: usize) -> Self {
        let bits_per_candidate = BitEncoder::num_bits(num_candidates + 1);
        assert!(bits_per_candidate <= 55);
        Self {
            bits_per_candidate,
            candidate_mask: !(!0 << bits_per_candidate),
            indices: Vec::new(),
            ballots: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn push(&mut self, ballot: Ballot) {
        self.indices.push(self.ballots.len());

        let mut encoder = BitEncoder::new(self.bits_per_candidate, &mut self.ballots);

        let ballot_count = ballot.count() as u64;
        assert!(ballot_count != 0);
        encoder.push_varint(ballot_count - 1);

        let has_tie = ballot.has_tie();
        encoder.push_bit(has_tie);
        for (i, rank) in ballot.order().enumerate() {
            // 1 bit = a new rank starts...
            if has_tie && i != 0 {
                encoder.push_bit(true);
            }
            for (j, &candidate) in rank.iter().enumerate() {
                // ...or the same rank continues.
                if has_tie && j != 0 {
                    encoder.push_bit(false);
                }
                // N bits = candidate id.
                encoder.push_candidate(candidate.into());
            }
        }

        // End marker (candidate = all 1s).
        encoder.finish(has_tie);
        encoder.flush();
    }

    pub fn count_allocations(&self, allocations: &mut BTreeMap<usize, usize>) {
        count_vec_allocations(allocations, &self.indices);
        count_vec_allocations(allocations, &self.ballots);
    }

    fn decode_at(&self, i: usize) -> Option<ExpandedBallot> {
        self.indices
            .get(i)
            .map(|&index| match self.bits_per_candidate {
                1 => BitDecoder::<1>::decode(&self.ballots, index),
                2 => BitDecoder::<2>::decode(&self.ballots, index),
                3 => BitDecoder::<3>::decode(&self.ballots, index),
                4 => BitDecoder::<4>::decode(&self.ballots, index),
                5 => BitDecoder::<5>::decode(&self.ballots, index),
                6 => BitDecoder::<6>::decode(&self.ballots, index),
                7 => BitDecoder::<7>::decode(&self.ballots, index),
                8 => BitDecoder::<8>::decode(&self.ballots, index),
                _ => panic!("At most 255 candidates are supported"),
            })
    }
}

struct BitEncoder<'a> {
    packed: u64,
    bit_count: u32,
    bits_per_candidate: u32,
    data: &'a mut Vec<u8>,
}

impl<'a> BitEncoder<'a> {
    fn new(bits_per_candidate: u32, data: &'a mut Vec<u8>) -> Self {
        Self {
            packed: 0,
            bit_count: 0,
            bits_per_candidate,
            data,
        }
    }

    fn num_bits(x: usize) -> u32 {
        if x <= 1 {
            0
        } else {
            (2 * x - 1).ilog2()
        }
    }

    fn push_bit(&mut self, bit: bool) {
        assert!(self.bit_count < 8);
        self.packed = (self.packed << 1) | (bit as u64);
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.data.push(self.packed as u8);
            self.bit_count = 0;
        }
    }

    fn push_byte(&mut self, byte: u8) {
        assert!(self.bit_count == 0);
        self.data.push(byte);
    }

    fn push_varint(&mut self, x: u64) {
        if x == 0 {
            self.push_bit(false);
        } else if x < 0x40 {
            // Fits 6 bits.
            self.push_byte(0x80 | (x as u8));
        } else if x < 0x2000 {
            // Fits 13 bits.
            self.push_byte(0xc0 | ((x >> 7) as u8));
            self.push_byte(x as u8 & 0x7f);
        } else if x < 0x10_0000 {
            // Fits 20 bits.
            self.push_byte(0xc0 | ((x >> 14) as u8));
            self.push_byte(0x80 | ((x >> 7) as u8));
            self.push_byte(x as u8 & 0x7f);
        } else if x < 0x800_0000 {
            // Fits 27 bits.
            self.push_byte(0xc0 | ((x >> 21) as u8));
            self.push_byte(0x80 | ((x >> 14) as u8));
            self.push_byte(0x80 | ((x >> 7) as u8));
            self.push_byte(x as u8 & 0x7f);
        } else if x < 0x4_0000_0000 {
            // Fits 34 bits.
            self.push_byte(0xc0 | ((x >> 28) as u8));
            self.push_byte(0x80 | ((x >> 21) as u8));
            self.push_byte(0x80 | ((x >> 14) as u8));
            self.push_byte(0x80 | ((x >> 7) as u8));
            self.push_byte(x as u8 & 0x7f);
        } else if x < 0x200_0000_0000 {
            // Fits 41 bits.
            self.push_byte(0xc0 | ((x >> 35) as u8));
            self.push_byte(0x80 | ((x >> 28) as u8));
            self.push_byte(0x80 | ((x >> 21) as u8));
            self.push_byte(0x80 | ((x >> 14) as u8));
            self.push_byte(0x80 | ((x >> 7) as u8));
            self.push_byte(x as u8 & 0x7f);
        } else if x < 0x1_0000_0000_0000 {
            // Fits 48 bits.
            self.push_byte(0xc0 | ((x >> 42) as u8));
            self.push_byte(0x80 | ((x >> 35) as u8));
            self.push_byte(0x80 | ((x >> 28) as u8));
            self.push_byte(0x80 | ((x >> 21) as u8));
            self.push_byte(0x80 | ((x >> 14) as u8));
            self.push_byte(0x80 | ((x >> 7) as u8));
            self.push_byte(x as u8 & 0x7f);
        } else if x < 0x80_0000_0000_0000 {
            // Fits 55 bits.
            self.push_byte(0xc0 | ((x >> 49) as u8));
            self.push_byte(0x80 | ((x >> 42) as u8));
            self.push_byte(0x80 | ((x >> 35) as u8));
            self.push_byte(0x80 | ((x >> 28) as u8));
            self.push_byte(0x80 | ((x >> 21) as u8));
            self.push_byte(0x80 | ((x >> 14) as u8));
            self.push_byte(0x80 | ((x >> 7) as u8));
            self.push_byte(x as u8 & 0x7f);
        } else if x < 0x4000_0000_0000_0000 {
            // Fits 62 bits.
            self.push_byte(0xc0 | ((x >> 56) as u8));
            self.push_byte(0x80 | ((x >> 49) as u8));
            self.push_byte(0x80 | ((x >> 42) as u8));
            self.push_byte(0x80 | ((x >> 35) as u8));
            self.push_byte(0x80 | ((x >> 28) as u8));
            self.push_byte(0x80 | ((x >> 21) as u8));
            self.push_byte(0x80 | ((x >> 14) as u8));
            self.push_byte(0x80 | ((x >> 7) as u8));
            self.push_byte(x as u8 & 0x7f);
        } else {
            self.push_byte(0xc0 | ((x >> 63) as u8));
            self.push_byte(0x80 | ((x >> 56) as u8));
            self.push_byte(0x80 | ((x >> 49) as u8));
            self.push_byte(0x80 | ((x >> 42) as u8));
            self.push_byte(0x80 | ((x >> 35) as u8));
            self.push_byte(0x80 | ((x >> 28) as u8));
            self.push_byte(0x80 | ((x >> 21) as u8));
            self.push_byte(0x80 | ((x >> 14) as u8));
            self.push_byte(0x80 | ((x >> 7) as u8));
            self.push_byte(x as u8 & 0x7f);
        }
    }

    fn push_fixed(&mut self, bits: u32, x: u64) {
        self.packed = (self.packed << bits) | x;
        self.bit_count += bits;
        while self.bit_count >= 8 {
            self.data.push((self.packed >> (self.bit_count - 8)) as u8);
            self.bit_count -= 8;
        }
    }

    fn push_candidate(&mut self, x: usize) {
        assert!(self.bit_count < 8);
        assert!(Self::num_bits(x) <= self.bits_per_candidate);
        self.push_fixed(self.bits_per_candidate, x as u64);
    }

    fn finish(&mut self, has_tie: bool) {
        assert!(self.bit_count < 8);
        let mut shift = self.bits_per_candidate;
        if has_tie {
            shift += 1;
        }
        self.push_fixed(shift, !(!0 << shift));
    }

    fn flush(self) {
        assert!(self.bit_count < 8);
        if self.bit_count > 0 {
            self.data.push((self.packed << (8 - self.bit_count)) as u8);
        }
    }
}

struct BitDecoder<'a, const BITS: u32> {
    packed: u64,
    bit_count: u32,
    data: &'a [u8],
    index: usize,
}

impl<'a, const BITS: u32> BitDecoder<'a, BITS> {
    const CANDIDATE_MASK: u64 = !(!0 << BITS);

    fn decode(data: &'a [u8], index: usize) -> ExpandedBallot {
        let mut decoder = Self::new(data, index);
        let count = (decoder.pop_varint() + 1) as usize;
        let has_tie = decoder.pop_bit();
        let mut order = [0; 64];
        let mut order_indices = [0; 64];
        let mut order_len = 0;
        let mut indices_len = 0;

        if has_tie {
            loop {
                if order_len == 0 || decoder.pop_bit() {
                    order_indices[indices_len] = order_len;
                    indices_len += 1;
                }
                let candidate = decoder.pop_candidate();
                if candidate == Self::CANDIDATE_MASK {
                    break;
                }
                order[order_len] = candidate as usize;
                order_len += 1;
            }
        } else {
            loop {
                order_indices[indices_len] = order_len;
                indices_len += 1;
                let candidate = decoder.pop_candidate();
                if candidate == Self::CANDIDATE_MASK {
                    break;
                }
                order[order_len] = candidate as usize;
                order_len += 1;
            }
        }

        ExpandedBallot {
            count,
            has_tie,
            order,
            order_indices,
            indices_len,
        }
    }

    fn new(data: &'a [u8], index: usize) -> Self {
        Self {
            packed: 0,
            bit_count: 0,
            data,
            index,
        }
    }

    fn fetch_byte(&mut self) {
        self.packed = (self.packed << 8) | (self.data[self.index] as u64);
        self.index += 1;
        self.bit_count += 8;
    }

    fn pop_bit(&mut self) -> bool {
        if self.bit_count == 0 {
            self.fetch_byte();
        }
        self.bit_count -= 1;
        (self.packed >> self.bit_count) & 1 != 0
    }

    fn pop_byte(&mut self) -> u8 {
        let byte = self.data[self.index];
        self.index += 1;
        byte
    }

    fn pop_candidate(&mut self) -> u64 {
        while self.bit_count < BITS {
            self.fetch_byte();
        }
        self.bit_count -= BITS;
        (self.packed >> self.bit_count) & Self::CANDIDATE_MASK
    }

    fn pop_varint(&mut self) -> u64 {
        self.fetch_byte();
        if self.packed & 0x80 == 0 {
            self.bit_count -= 1;
            0
        } else {
            self.bit_count -= 8;
            let mut result = self.packed & 0x3f;
            if self.packed & 0x40 == 0 {
                result
            } else {
                loop {
                    let byte = self.pop_byte() as u64;
                    result = (result << 7) | (byte & 0x7f);
                    if byte & 0x80 == 0 {
                        return result;
                    }
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ExpandedBallot {
    count: usize,
    has_tie: bool,
    //order: SmallVec<[usize; 64]>,
    //order_indices: SmallVec<[usize; 64]>,
    order: [usize; 64],
    order_indices: [usize; 64],
    indices_len: usize,
}

#[cfg(test)]
impl ExpandedBallot {
    fn new(ballot: Ballot) -> Self {
        //let mut order = SmallVec::new();
        //let mut order_indices = SmallVec::new();
        //order_indices.push(0);
        let mut order = [0; 64];
        let mut order_indices = [0; 64];
        let mut order_len = 0;
        let mut indices_len = 1;

        for rank in ballot.order() {
            for &x in rank {
                //order.push(x.into());
                order[order_len] = x.into();
                order_len += 1;
            }
            //order_indices.push(order.len());
            order_indices[indices_len] = order_len;
            indices_len += 1;
        }

        if order_len == 0 {
            order_indices = [0; 64];
            indices_len = 0;
        }
        /*
        if order.is_empty() {
            order_indices = SmallVec::new();
        }
        */

        Self {
            count: ballot.count(),
            has_tie: ballot.has_tie(),
            order,
            order_indices,
            indices_len,
        }
    }
}

impl BallotView for ExpandedBallot {
    fn count(&self) -> usize {
        self.count
    }

    fn has_tie(&self) -> bool {
        self.has_tie
    }

    fn order(&self) -> impl Iterator<Item = &[impl Into<usize> + Copy + '_]> + '_ {
        //self.order_indices
        self.order_indices[..self.indices_len]
            .array_windows()
            .map(|&[start, end]| &self.order[start..end])
    }

    fn order_len(&self) -> usize {
        //self.order_indices.len() - 1
        self.indices_len - 1
    }

    fn order_at(&self, i: usize) -> &[impl Into<usize> + Copy + '_] {
        let start = self.order_indices[i];
        let end = self.order_indices[i + 1];
        &self.order[start..end]
    }
}

// Cursor.
pub(crate) struct BitPackedBallotsCursor<'a> {
    ballots: &'a BitPackedBallots,
    range: std::ops::Range<usize>,
}

impl<'a> BitPackedBallotsCursor<'a> {
    pub fn new(ballots: &'a BitPackedBallots) -> Self {
        Self {
            ballots,
            range: 0..ballots.len(),
        }
    }
}

impl<'a> Iterator for BitPackedBallotsCursor<'a> {
    type Item = ExpandedBallot;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.range.is_empty() {
            let result = self.ballots.decode_at(self.range.start);
            self.range.start += 1;
            result
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.range.end - self.range.start;
        (size, Some(size))
    }
}

impl<'a> ExactSizeIterator for BitPackedBallotsCursor<'a> {}

impl<'a> DoubleEndedIterator for BitPackedBallotsCursor<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if !self.range.is_empty() {
            let result = self.ballots.decode_at(self.range.end - 1);
            self.range.end -= 1;
            result
        } else {
            None
        }
    }
}

impl<'a> IntoParallelIterator for BitPackedBallotsCursor<'a> {
    type Item = ExpandedBallot;
    type Iter = &'a BitPackedBallots;

    fn into_par_iter(self) -> Self::Iter {
        self.ballots
    }
}

impl<'a> BallotCursor for BitPackedBallotsCursor<'a> {
    type B = ExpandedBallot;
    type I = &'a BitPackedBallots;

    fn at(&self, index: usize) -> Option<Self::B> {
        if self.range.contains(&index) {
            self.ballots.decode_at(index)
        } else {
            None
        }
    }
}

// Rayon integration.
impl<'a> ParallelIterator for &'a BitPackedBallots {
    type Item = ExpandedBallot;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<'a> IndexedParallelIterator for &'a BitPackedBallots {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        (*self).len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(BitPackedBallotsCursor {
            ballots: self,
            range: 0..self.len(),
        })
    }
}

impl<'a> Producer for BitPackedBallotsCursor<'a> {
    type Item = ExpandedBallot;
    type IntoIter = Self;

    fn into_iter(self) -> Self::IntoIter {
        self
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.range.start + index;
        let left = self.range.start..mid;
        let right = mid..self.range.end;
        (
            Self {
                ballots: self.ballots,
                range: left,
            },
            Self {
                ballots: self.ballots,
                range: right,
            },
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[allow(clippy::unusual_byte_groupings)]
    #[test]
    fn test_encode() {
        let mut ballots = BitPackedBallots::new(10);
        ballots.push(Ballot::new(1, [vec![1], vec![2], vec![3]]));
        ballots.push(Ballot::new(1, [vec![1, 2], vec![3]]));
        ballots.push(Ballot::new(42, [vec![1], vec![2], vec![3]]));
        assert_eq!(
            ballots,
            BitPackedBallots {
                bits_per_candidate: 4,
                candidate_mask: 0x0F,
                indices: vec![0, 3, 6],
                ballots: vec![
                    // Ballot 1
                    0b0_0_0001_00,
                    0b10_0011_11,
                    0b11_000000,
                    // Ballot 2
                    0b0_1_0001_0_0,
                    0b010_1_0011,
                    0b1_1111_000,
                    // Ballot 3
                    0b1_0101001,
                    0b0_0001_001,
                    0b0_0011_111,
                    0b1_0000000,
                ],
            }
        );
    }

    #[test]
    fn test_encode_decode() {
        let mut ballots = BitPackedBallots::new(10);
        ballots.push(Ballot::new(1, [vec![1], vec![2], vec![3]]));
        ballots.push(Ballot::new(1, [vec![1, 2], vec![3]]));
        ballots.push(Ballot::new(42, [vec![1], vec![2], vec![3]]));

        assert_eq!(
            [
                ballots.decode_at(0),
                ballots.decode_at(1),
                ballots.decode_at(2),
                ballots.decode_at(3),
            ],
            [
                Some(ExpandedBallot::new(Ballot::new(
                    1,
                    [vec![1], vec![2], vec![3]]
                ))),
                Some(ExpandedBallot::new(Ballot::new(1, [vec![1, 2], vec![3]]))),
                Some(ExpandedBallot::new(Ballot::new(
                    42,
                    [vec![1], vec![2], vec![3]]
                ))),
                None
            ]
        );
    }
}
