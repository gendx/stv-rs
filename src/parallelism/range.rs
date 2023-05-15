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

use log::{debug, trace, warn};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A factory for handing out ranges of items to various threads.
pub trait RangeFactory {
    type Rn: Range;
    type Orchestrator: RangeOrchestrator;

    /// Creates a new factory for a range with the given number of elements
    /// split across the given number of threads.
    fn new(num_elements: usize, num_threads: usize) -> Self;

    /// Returns the orchestrator object for all the ranges created by this
    /// factory.
    fn orchestrator(self) -> Self::Orchestrator;

    /// Returns the range for the given thread.
    fn range(&self, thread_id: usize) -> Self::Rn;
}

/// An orchestrator for the ranges given to all the threads.
pub trait RangeOrchestrator {
    /// Resets all the ranges to prepare a new computation round.
    fn reset_ranges(&self);
}

/// A range of items similar to [`std::ops::Range`], but that can steal from or
/// be stolen by other threads.
pub trait Range {
    type Iter: Iterator<Item = usize>;

    /// Returns an iterator over the items in this range. The item can be
    /// dynamically stolen from/by other threads, but the iterator provides
    /// a safe abstraction over that.
    fn iter(&self) -> Self::Iter;
}

/// A factory that hands out a fixed range to each thread, without any stealing.
pub struct FixedRangeFactory {
    /// Total number of elements to iterate over.
    num_elements: usize,
    /// Number of threads that iterate.
    num_threads: usize,
}

impl RangeFactory for FixedRangeFactory {
    type Rn = FixedRange;
    type Orchestrator = FixedRangeOrchestrator;

    fn new(num_elements: usize, num_threads: usize) -> Self {
        Self {
            num_elements,
            num_threads,
        }
    }

    fn orchestrator(self) -> FixedRangeOrchestrator {
        FixedRangeOrchestrator {}
    }

    fn range(&self, thread_id: usize) -> FixedRange {
        let start = (thread_id * self.num_elements) / self.num_threads;
        let end = ((thread_id + 1) * self.num_elements) / self.num_threads;
        FixedRange(start..end)
    }
}

/// An orchestrator for the [`FixedRangeFactory`].
pub struct FixedRangeOrchestrator {}

impl RangeOrchestrator for FixedRangeOrchestrator {
    fn reset_ranges(&self) {
        // Nothing to do.
    }
}

/// A fixed range.
#[derive(Debug, PartialEq, Eq)]
pub struct FixedRange(std::ops::Range<usize>);

impl Range for FixedRange {
    type Iter = std::ops::Range<usize>;

    fn iter(&self) -> Self::Iter {
        self.0.clone()
    }
}

/// A factory for ranges that implement work stealing among threads.
///
/// Whenever a thread finishes processing its range, it looks for another range
/// to steal from. It then divides that range into two and steals a half, to
/// continue processing items.
pub struct WorkStealingRangeFactory {
    /// Total number of elements to iterate over.
    num_elements: usize,
    /// Handle to the ranges of all the threads.
    ranges: Arc<Vec<AtomicRange>>,
}

impl RangeFactory for WorkStealingRangeFactory {
    type Rn = WorkStealingRange;
    type Orchestrator = WorkStealingRangeOrchestrator;

    fn new(num_elements: usize, num_threads: usize) -> Self {
        Self {
            num_elements,
            ranges: Arc::new((0..num_threads).map(|_| AtomicRange::default()).collect()),
        }
    }

    fn orchestrator(self) -> WorkStealingRangeOrchestrator {
        WorkStealingRangeOrchestrator {
            num_elements: self.num_elements,
            ranges: self.ranges,
        }
    }

    fn range(&self, thread_id: usize) -> WorkStealingRange {
        WorkStealingRange {
            id: thread_id,
            ranges: self.ranges.clone(),
        }
    }
}

/// An orchestrator for the [`WorkStealingRangeFactory`].
pub struct WorkStealingRangeOrchestrator {
    /// Total number of elements to iterate over.
    num_elements: usize,
    /// Handle to the ranges of all the threads.
    ranges: Arc<Vec<AtomicRange>>,
}

impl RangeOrchestrator for WorkStealingRangeOrchestrator {
    fn reset_ranges(&self) {
        debug!("Resetting ranges.");
        let num_threads = self.ranges.len();
        for (i, range) in self.ranges.iter().enumerate() {
            let start = (i * self.num_elements) / num_threads;
            let end = ((i + 1) * self.num_elements) / num_threads;
            range.store(PackedRange::new(start as u32, end as u32));
        }
    }
}

/// A range that implements work stealing.
pub struct WorkStealingRange {
    /// Index of the thread that owns this range.
    id: usize,
    /// Handle to the ranges of all the threads.
    ranges: Arc<Vec<AtomicRange>>,
}

impl Range for WorkStealingRange {
    type Iter = WorkStealingRangeIterator;

    fn iter(&self) -> Self::Iter {
        WorkStealingRangeIterator {
            id: self.id,
            ranges: self.ranges.clone(),
        }
    }
}

/// A [start, end) pair that can atomically be modified.
struct AtomicRange(AtomicU64);

impl Default for AtomicRange {
    #[inline(always)]
    fn default() -> Self {
        AtomicRange::new(PackedRange::default())
    }
}

impl AtomicRange {
    /// Creates a new atomic range.
    #[inline(always)]
    fn new(range: PackedRange) -> Self {
        AtomicRange(AtomicU64::new(range.0))
    }

    /// Atomically loads the range.
    #[inline(always)]
    fn load(&self) -> PackedRange {
        PackedRange(self.0.load(Ordering::SeqCst))
    }

    /// Atomically stores the range.
    #[inline(always)]
    fn store(&self, range: PackedRange) {
        self.0.store(range.0, Ordering::SeqCst)
    }

    /// Atomically compares and exchanges the range. In case of failure, the
    /// range contained in the atomic variable is returned.
    #[inline(always)]
    fn compare_exchange(&self, before: PackedRange, after: PackedRange) -> Result<(), PackedRange> {
        match self
            .0
            .compare_exchange(before.0, after.0, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => Ok(()),
            Err(e) => Err(PackedRange(e)),
        }
    }
}

/// A [start, end) range that fits into a `u64`, and can therefore be
/// loaded/stored atomically.
#[derive(Clone, Copy, Default)]
struct PackedRange(u64);

impl PackedRange {
    /// Creates a range with the given [start, end) pair.
    #[inline(always)]
    fn new(start: u32, end: u32) -> Self {
        Self((start as u64) | ((end as u64) << 32))
    }

    /// Reads the start of the range (inclusive).
    #[inline(always)]
    fn start(self) -> u32 {
        self.0 as u32
    }

    /// Reads the end of the range (exclusive).
    #[inline(always)]
    fn end(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Increments the start of the range.
    #[inline(always)]
    fn increment_start(self) -> Self {
        assert!(self.start() < self.end());
        // TODO: check for overflow.
        PackedRange::new(self.start() + 1, self.end())
    }

    /// Splits the range into two halves. If the input range is non-empty, the
    /// second half is guaranteed to be non-empty.
    #[inline(always)]
    fn split(self) -> (Self, Self) {
        let start = self.start();
        let end = self.end();
        // TODO: check for overflow.
        let middle = (start + end) / 2;
        (
            PackedRange::new(start, middle),
            PackedRange::new(middle, end),
        )
    }

    /// Checks if the range is empty.
    #[inline(always)]
    fn is_empty(self) -> bool {
        self.start() == self.end()
    }
}

/// An iterator for the [`WorkStealingRange`].
pub struct WorkStealingRangeIterator {
    /// Index of the thread that owns this range.
    id: usize,
    /// Handle to the ranges of all the threads.
    ranges: Arc<Vec<AtomicRange>>,
}

impl Iterator for WorkStealingRangeIterator {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let my_atomic_range: &AtomicRange = &self.ranges[self.id];
        let mut my_range: PackedRange = my_atomic_range.load();
        loop {
            if !my_range.is_empty() {
                let my_new_range = my_range.increment_start();
                match my_atomic_range.compare_exchange(my_range, my_new_range) {
                    Ok(()) => {
                        trace!(
                            "[thread {}] Incremented range to {}..{}.",
                            self.id,
                            my_new_range.start(),
                            my_new_range.end()
                        );
                        return Some(my_range.start() as usize);
                    }
                    Err(range) => {
                        my_range = range;
                        warn!(
                            "[thread {}] Failed to increment range, new range is {}..{}.",
                            self.id,
                            range.start(),
                            range.end()
                        );
                        continue;
                    }
                }
            } else {
                debug!(
                    "[thread {}] Range {}..{} is empty, scanning other threads.",
                    self.id,
                    my_range.start(),
                    my_range.end()
                );
                let range_count = self.ranges.len();
                // TODO: don't necessarily steal in order
                'others: for i in 1..range_count {
                    let index: usize = (self.id + i) % range_count;
                    let other_atomic_range: &AtomicRange = &self.ranges[index];
                    let mut other_range = other_atomic_range.load();
                    'inner: loop {
                        if other_range.is_empty() {
                            continue 'others;
                        }

                        // Steal some work.
                        let (remaining, stolen) = other_range.split();
                        match other_atomic_range.compare_exchange(other_range, remaining) {
                            Ok(()) => {
                                my_atomic_range.store(stolen.increment_start());
                                debug!(
                                    "[thread {}] Stole from #{index}: stole {}..{} and left {}..{}.",
                                    self.id,
                                    stolen.start(),
                                    stolen.end(),
                                    remaining.start(),
                                    remaining.end()
                                );
                                return Some(stolen.start() as usize);
                            }
                            Err(range) => {
                                other_range = range;
                                warn!(
                                    "[thread {}] Failed to steal from #{index}, new range is {}..{}.",
                                    self.id,
                                    range.start(),
                                    range.end()
                                );
                                continue 'inner;
                            }
                        }
                    }
                }
                debug!("[thread {}] Didn't find anything to steal", self.id);
                // Didn't manage to steal anything.
                return None;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_fixed_range_factory_splits_evenly() {
        let factory = FixedRangeFactory::new(100, 4);
        assert_eq!(factory.range(0), FixedRange(0..25));
        assert_eq!(factory.range(1), FixedRange(25..50));
        assert_eq!(factory.range(2), FixedRange(50..75));
        assert_eq!(factory.range(3), FixedRange(75..100));

        let factory = FixedRangeFactory::new(100, 7);
        assert_eq!(factory.range(0), FixedRange(0..14));
        assert_eq!(factory.range(1), FixedRange(14..28));
        assert_eq!(factory.range(2), FixedRange(28..42));
        assert_eq!(factory.range(3), FixedRange(42..57));
        assert_eq!(factory.range(4), FixedRange(57..71));
        assert_eq!(factory.range(5), FixedRange(71..85));
        assert_eq!(factory.range(6), FixedRange(85..100));
    }

    #[test]
    fn test_fixed_range() {
        let factory = FixedRangeFactory::new(100, 4);
        let ranges: [_; 4] = std::array::from_fn(|i| factory.range(i));
        let orchestrator = factory.orchestrator();

        std::thread::scope(|s| {
            for _ in 0..10 {
                orchestrator.reset_ranges();
                let handles = ranges
                    .each_ref()
                    .map(|range| s.spawn(move || range.iter().collect::<Vec<_>>()));
                let values: [Vec<usize>; 4] = handles.map(|handle| handle.join().unwrap());

                // The fixed range implementation always yields the same items in order.
                for (i, set) in values.iter().enumerate() {
                    assert_eq!(*set, (i * 25..(i + 1) * 25).collect::<Vec<_>>());
                }
            }
        });
    }

    #[test]
    fn test_work_stealing_range() {
        const NUM_THREADS: usize = 4;
        const NUM_ELEMENTS: usize = 10000;

        let factory = WorkStealingRangeFactory::new(NUM_ELEMENTS, NUM_THREADS);
        let ranges: [_; NUM_THREADS] = std::array::from_fn(|i| factory.range(i));
        let orchestrator = factory.orchestrator();

        std::thread::scope(|s| {
            for _ in 0..10 {
                orchestrator.reset_ranges();
                let handles = ranges
                    .each_ref()
                    .map(|range| s.spawn(move || range.iter().collect::<Vec<_>>()));
                let values: [Vec<usize>; NUM_THREADS] =
                    handles.map(|handle| handle.join().unwrap());

                // This checks that:
                // - all ranges yield disjoint elements,
                // - each range never yields the same element twice.
                let mut all_values = vec![false; NUM_ELEMENTS];
                for set in values {
                    println!("Values: {set:?}");
                    for x in set {
                        assert!(!all_values[x]);
                        all_values[x] = true;
                    }
                }
                // Check that the whole range is covered.
                assert!(all_values.iter().all(|x| *x));
            }
        });
    }

    #[test]
    fn test_default_packed_range_is_empty() {
        let range = PackedRange::default();
        assert!(range.is_empty());
        assert_eq!(range.start(), 0);
        assert_eq!(range.end(), 0);
    }

    #[test]
    fn test_packed_range_is_consistent() {
        for i in 0..30 {
            for j in i..30 {
                let range = PackedRange::new(i, j);
                assert_eq!(range.start(), i);
                assert_eq!(range.end(), j);
            }
        }
    }

    #[test]
    fn test_packed_range_increment_start() {
        let mut range = PackedRange::new(0, 10);

        for i in 1..=10 {
            range = range.increment_start();
            assert_eq!((range.start(), range.end()), (i, 10));
        }
    }

    #[test]
    fn test_packed_range_split() {
        let (left, right) = PackedRange::new(0, 0).split();
        assert!(left.is_empty());
        assert_eq!((left.start(), left.end()), (0, 0));
        assert!(right.is_empty());
        assert_eq!((right.start(), right.end()), (0, 0));

        let (left, right) = PackedRange::new(0, 1).split();
        assert!(left.is_empty());
        assert_eq!((left.start(), left.end()), (0, 0));
        assert!(!right.is_empty());
        assert_eq!((right.start(), right.end()), (0, 1));
    }

    #[test]
    fn test_packed_range_split_is_exhaustive() {
        for i in 0..100 {
            for j in i..100 {
                let (left, right) = PackedRange::new(i, j).split();
                assert!(left.start() <= left.end());
                assert!(right.start() <= right.end());
                assert_eq!(left.start(), i);
                assert_eq!(left.end(), right.start());
                assert_eq!(right.end(), j);
            }
        }
    }

    #[test]
    fn test_packed_range_split_is_fair() {
        for i in 0..100 {
            for j in i..100 {
                let (left, right) = PackedRange::new(i, j).split();
                assert!(left.end() - left.start() <= right.end() - right.start());
                assert!(right.end() - right.start() <= left.end() - left.start() + 1);
                if i != j {
                    assert!(!right.is_empty());
                }
            }
        }
    }
}
