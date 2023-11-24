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

//! Hand-rolled parallelism utilities for vote counting.

mod range;
mod thread_pool;

pub use thread_pool::{RangeStrategy, ThreadAccumulator, ThreadPool};

#[cfg(test)]
mod test {
    use super::*;
    use std::num::NonZeroUsize;

    /// Example of accumulator that computes a sum of integers.
    struct SumAccumulator;

    impl ThreadAccumulator<u64, u64> for SumAccumulator {
        type Accumulator<'a> = u64;

        fn init(&self) -> u64 {
            0
        }

        fn process_item(&self, accumulator: &mut u64, _index: usize, x: &u64) {
            *accumulator += *x;
        }

        fn finalize(&self, accumulator: u64) -> u64 {
            accumulator
        }
    }

    macro_rules! parallelism_tests {
        ( $mod:ident, $range_strategy:expr, $($case:ident,)+ ) => {
            mod $mod {
                use super::*;

                $(
                #[test]
                fn $case() {
                    $crate::parallelism::test::$case($range_strategy);
                }
                )+
            }
        };
    }

    macro_rules! all_parallelism_tests {
        ( $mod:ident, $range_strategy:expr ) => {
            parallelism_tests!($mod, $range_strategy, test_sum_integers, test_sum_twice,);
        };
    }

    all_parallelism_tests!(fixed, RangeStrategy::Fixed);
    all_parallelism_tests!(work_stealing, RangeStrategy::WorkStealing);

    fn test_sum_integers(range_strategy: RangeStrategy) {
        let input = (0..=10_000).collect::<Vec<u64>>();
        let num_threads = NonZeroUsize::try_from(4).unwrap();
        let sum = std::thread::scope(|scope| {
            let thread_pool = ThreadPool::new(scope, num_threads, range_strategy, &input, || {
                SumAccumulator
            });
            thread_pool.process_inputs().reduce(|a, b| a + b).unwrap()
        });
        assert_eq!(sum, 5_000 * 10_001);
    }

    fn test_sum_twice(range_strategy: RangeStrategy) {
        let input = (0..=10_000).collect::<Vec<u64>>();
        let num_threads = NonZeroUsize::try_from(4).unwrap();
        let (sum1, sum2) = std::thread::scope(|scope| {
            let thread_pool = ThreadPool::new(scope, num_threads, range_strategy, &input, || {
                SumAccumulator
            });
            // The same input can be processed multiple times on the thread pool.
            let sum1 = thread_pool.process_inputs().reduce(|a, b| a + b).unwrap();
            let sum2 = thread_pool.process_inputs().reduce(|a, b| a + b).unwrap();
            (sum1, sum2)
        });
        assert_eq!(sum1, 5_000 * 10_001);
        assert_eq!(sum2, 5_000 * 10_001);
    }
}
