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

use super::util::count_vec_allocations;
use std::collections::BTreeMap;
use std::ops::Deref;

/// Ballot cast in the election.
pub type Ballot = BallotImpl<VecOrder<usize>>;

/// Ballot cast in the election.
#[derive(Debug, PartialEq, Eq)]
pub struct BallotImpl<O: Order> {
    /// Number of electors that have cast this ballot.
    count: usize,
    /// Ordering of candidates in this ballot.
    order: O,
}

impl<O: Order> BallotImpl<O> {
    /// Constructs a new ballot.
    #[inline(always)]
    pub fn new(
        count: usize,
        order: impl IntoIterator<Item = impl IntoIterator<Item = usize>>,
    ) -> Self {
        Self {
            count,
            order: O::new(order),
        }
    }

    /// Returns an empty ballot with a count of zero.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self {
            count: 0,
            order: O::empty(),
        }
    }

    /// Returns an empty ballot with the given count.
    #[cfg(test)]
    pub(crate) fn empties(count: usize) -> Self {
        Self {
            count,
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
    #[inline(always)]
    pub(crate) fn order_len(&self) -> usize {
        self.order.len()
    }

    /// Returns the rank at the given index in the ballot order.
    #[inline(always)]
    pub(crate) fn order_at(&self, i: usize) -> &[impl Into<usize> + Copy + '_] {
        self.order.at(i)
    }

    /// Returns whether this ballot contains candidates ranked equally.
    #[inline(always)]
    pub fn has_tie(&self) -> bool {
        self.order().any(|ranking| ranking.len() != 1)
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

    #[inline(always)]
    pub(super) fn count_allocations(&self, allocations: &mut BTreeMap<usize, usize>) {
        self.order.count_allocations(allocations)
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
}

/// Ordering of candidates in a ballot.
#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(test, derive(Clone))]
pub struct VecOrder<T> {
    /// The outer [`Vec`] represents the ranking of candidates, from most
    /// preferred to least preferred. The inner [`Vec`] represents candidates
    /// ranked equally at a given order.
    order: Vec<Vec<T>>,
}

impl<T> Order for VecOrder<T>
where
    T: TryFrom<usize> + Into<usize> + Copy,
{
    #[inline(always)]
    fn new(order: impl IntoIterator<Item = impl IntoIterator<Item = usize>>) -> Self {
        Self {
            order: order
                .into_iter()
                .map(|rank| {
                    rank.into_iter()
                        .map(|x| x.try_into().ok().unwrap())
                        .collect()
                })
                .collect(),
        }
    }

    #[cfg(test)]
    fn empty() -> Self {
        Self { order: Vec::new() }
    }

    #[inline(always)]
    fn order(&self) -> impl Iterator<Item = &[impl Into<usize> + Copy]> {
        self.order.iter().map(|x| x.deref())
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.order.len()
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    #[inline(always)]
    fn at(&self, i: usize) -> &[impl Into<usize> + Copy] {
        &self.order[i]
    }

    #[inline(always)]
    fn candidates(&self) -> impl Iterator<Item = &(impl Into<usize> + Copy)> {
        self.order.iter().flatten()
    }

    #[inline(always)]
    fn count_allocations(&self, allocations: &mut BTreeMap<usize, usize>) {
        count_vec_allocations(allocations, &self.order);
        for rank in &self.order {
            count_vec_allocations(allocations, rank);
        }
    }
}
