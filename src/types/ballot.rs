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

//! Types to represent ballots in an election.

use super::util::{address_of_slice, count_slice_allocations};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use std::collections::BTreeMap;
use std::fmt::Debug;

/// A cursor over a sequence of ballots, allowing serial and parallel iteration.
pub trait BallotCursor:
    Iterator<Item = Self::B> + IntoParallelIterator<Item = Self::B, Iter = Self::I>
{
    /// View over one ballot.
    type B: BallotView;
    /// Rayon's indexed parallel iterator over the ballots.
    type I: IndexedParallelIterator<Item = Self::B>;

    /// Returns the ballot at the given index.
    fn at(&self, index: usize) -> Option<Self::B>;
}

/// View over one ballot.
pub trait BallotView: Debug {
    /// Number of electors that have cast this ballot.
    fn count(&self) -> usize;

    /// Whether this ballot contains candidates ranked equally.
    fn has_tie(&self) -> bool;

    /// Returns the order of candidates in the ballot. The iterator yields
    /// candidates from most preferred to least preferred. Each item
    /// contains a set of candidates ranked equally.
    fn order(&self) -> impl Iterator<Item = &[impl Into<usize> + Copy + '_]> + '_;

    /// Returns the number of successive ranks in the ballot order.
    fn order_len(&self) -> usize;

    /// Returns the rank at the given index in the ballot order.
    fn order_at(&self, i: usize) -> &[impl Into<usize> + Copy + '_];
}

#[cfg(test)]
impl BallotView for &Ballot {
    fn count(&self) -> usize {
        (*self).count()
    }

    fn has_tie(&self) -> bool {
        (*self).has_tie()
    }

    fn order(&self) -> impl Iterator<Item = &[impl Into<usize> + Copy + '_]> + '_ {
        (*self).order()
    }

    fn order_len(&self) -> usize {
        (*self).order_len()
    }

    fn order_at(&self, i: usize) -> &[impl Into<usize> + Copy + '_] {
        (*self).order_at(i)
    }
}

/// Ballot cast in the election.
pub type Ballot = BallotImpl<BoxedFlatOrder>;

/// Ballot cast in the election.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BallotImpl<O: Order + Clone> {
    /// Number of electors that have cast this ballot.
    count: usize,
    /// Whether this ballot contains candidates ranked equally.
    has_tie: bool,
    /// Ordering of candidates in this ballot.
    order: O,
}

impl<O: Order + Clone, B: BallotView> From<B> for BallotImpl<O> {
    fn from(ballot: B) -> Self {
        Self::new(
            ballot.count(),
            ballot.order().map(|rank| rank.iter().map(|&x| x.into())),
        )
    }
}

impl<O: Order + Clone> BallotImpl<O> {
    /// Constructs a new ballot.
    #[inline(always)]
    pub fn new(
        count: usize,
        order: impl IntoIterator<Item = impl IntoIterator<Item = usize>>,
    ) -> Self {
        let order = O::new(order);
        let has_tie = order.order().any(|ranking| ranking.len() != 1);
        Self {
            count,
            has_tie,
            order,
        }
    }

    /// Returns an empty ballot with the given count.
    #[cfg(test)]
    pub(crate) fn empties(count: usize) -> Self {
        Self {
            count,
            has_tie: false,
            order: O::empty(),
        }
    }

    /// Returns the number of times this ballot was cast.
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the order of candidates in the ballot. The iterator yields
    /// candidates from most preferred to least preferred. Each item
    /// contains a set of candidates ranked equally.
    #[inline(always)]
    pub fn order(&self) -> impl Iterator<Item = &[impl Into<usize> + Copy + '_]> + '_ {
        self.order.order()
    }

    /// Returns whether this ballot is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Returns the number of successive ranks in the ballot order.
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn order_len(&self) -> usize {
        self.order.len()
    }

    /// Returns the rank at the given index in the ballot order.
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn order_at(&self, i: usize) -> &[impl Into<usize> + Copy + '_] {
        self.order.at(i)
    }

    /// Returns whether this ballot contains candidates ranked equally.
    #[inline(always)]
    pub fn has_tie(&self) -> bool {
        self.has_tie
    }

    /// Checks that a ballot is valid, i.e. that no candidate appears twice in
    /// the ballot.
    pub fn validate(&self) {
        assert!(self.count() > 0);
        let mut all: Vec<usize> = self.candidates().map(|&x| x.into()).collect();
        all.sort_unstable();
        let len = all.len();
        all.dedup();
        assert_eq!(len, all.len());
    }

    /// Returns the set of candidates present in this ballot.
    #[inline(always)]
    fn candidates(&self) -> impl Iterator<Item = &(impl Into<usize> + Copy + '_)> + '_ {
        self.order.candidates()
    }
}

/// Ordering of candidates in a ballot.
pub trait Order {
    /// Constructs a new ballot order.
    fn new(order: impl IntoIterator<Item = impl IntoIterator<Item = usize>>) -> Self;

    /// Constructs an empty ballot order.
    #[cfg(test)]
    fn empty() -> Self;

    /// Returns an iterator over the ranks in this ballot order. Each item
    /// contains the set of candidates at that rank.
    fn order(&self) -> impl Iterator<Item = &[impl Into<usize> + Copy]>;

    /// Returns the number of ranks.
    fn len(&self) -> usize;

    /// Returns whether this order is empty.
    fn is_empty(&self) -> bool;

    /// Returns the rank at a given index.
    fn at(&self, i: usize) -> &[impl Into<usize> + Copy];

    /// Returns an iterator over all the candidates present in this ballot
    /// order.
    fn candidates(&self) -> impl Iterator<Item = &(impl Into<usize> + Copy)>;

    /// Counts the allocations used by this type, reporting them into the given
    /// map.
    fn count_allocations(&self, allocations: &mut BTreeMap<usize, usize>);

    /// Returns the addresses of heap-allocated items contained in this object.
    fn allocated_addresses(&self) -> impl Iterator<Item = usize>;
}

/// Ordering of candidates in a ballot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxedFlatOrder {
    /// Flattened list of all the candidates in the ballot.
    order: Box<[u8]>,
    /// Indices where each rank starts in this order.
    order_indices: Box<[u8]>,
}

impl Order for BoxedFlatOrder {
    fn new(into_order: impl IntoIterator<Item = impl IntoIterator<Item = usize>>) -> Self {
        let mut order = Vec::new();
        let mut order_indices = vec![0];
        for rank in into_order.into_iter() {
            for x in rank.into_iter() {
                order.push(x.try_into().unwrap());
            }
            order_indices.push(order.len().try_into().unwrap());
        }

        if order.is_empty() {
            order_indices = Vec::new();
        }

        Self {
            order: order.into_boxed_slice(),
            order_indices: order_indices.into_boxed_slice(),
        }
    }

    #[cfg(test)]
    fn empty() -> Self {
        Self {
            order: Box::new([]),
            order_indices: Box::new([]),
        }
    }

    #[inline(always)]
    fn order(&self) -> impl Iterator<Item = &[impl Into<usize> + Copy]> {
        // TODO: Use array_windows once stable.
        let count = if self.order_indices.is_empty() {
            0
        } else {
            self.order_indices.len() - 1
        };
        (0..count).map(|i| {
            let start = self.order_indices[i];
            let end = self.order_indices[i + 1];
            &self.order[start.into()..end.into()]
        })
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.order_indices.len() - 1
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.order_indices.is_empty()
    }

    #[inline(always)]
    fn at(&self, i: usize) -> &[impl Into<usize> + Copy] {
        let start = self.order_indices[i];
        let end = self.order_indices[i + 1];
        &self.order[start.into()..end.into()]
    }

    #[inline(always)]
    fn candidates(&self) -> impl Iterator<Item = &(impl Into<usize> + Copy)> {
        self.order.iter()
    }

    #[inline(always)]
    fn count_allocations(&self, allocations: &mut BTreeMap<usize, usize>) {
        count_slice_allocations(allocations, &self.order);
        count_slice_allocations(allocations, &self.order_indices);
    }

    #[inline(always)]
    fn allocated_addresses(&self) -> impl Iterator<Item = usize> {
        [
            address_of_slice(&self.order),
            address_of_slice(&self.order_indices),
        ]
        .into_iter()
        .flatten()
    }
}
