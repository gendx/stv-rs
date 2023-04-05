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

//! A hand-rolled thread pool, customized for the vote counting problem.

use crate::arithmetic::{Integer, IntegerRef, Rational, RationalRef};
use crate::types::Election;
use crate::vote_count::{VoteAccumulator, VoteCount};
use log::debug;
use std::cell::Cell;
use std::num::NonZeroUsize;
use std::ops::{DerefMut, Range};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{Scope, ScopedJoinHandle};

/// Status of the main thread.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MainStatus {
    /// The main thread is waiting for the worker threads to finish a round.
    Waiting,
    /// The main thread is ready to prepare the next round.
    Ready,
}

/// Status sent to the worker threads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkerStatus {
    /// The threads need to compute a vote counting round of the given color.
    Round(RoundColor),
    /// There is nothing more to do and the threads must exit.
    Finished,
}

/// An 2-element enumeration to distinguish successive rounds. The "colors" are
/// only illustrative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundColor {
    Blue,
    Red,
}

impl RoundColor {
    /// Flips to the other color.
    fn toggle(&mut self) {
        *self = match self {
            RoundColor::Blue => RoundColor::Red,
            RoundColor::Red => RoundColor::Blue,
        }
    }
}

/// A thread pool tied to a scope, that can perform vote counting rounds.
pub struct ThreadPool<'scope, I, R> {
    /// Handles to all the threads in the pool.
    threads: Vec<Thread<'scope, I, R>>,
    /// Number of worker threads active in the current round.
    num_active_threads: Arc<AtomicUsize>,
    /// Color of the current round.
    round: Cell<RoundColor>,
    /// Status of the worker threads.
    worker_status: Arc<(Mutex<WorkerStatus>, Condvar)>,
    /// Status of the main thread.
    main_status: Arc<(Mutex<MainStatus>, Condvar)>,
    /// Storage for the keep factors, used as input of the current round by the
    /// worker threads.
    keep_factors: Arc<RwLock<Vec<R>>>,
}

/// Handle to a thread in the pool.
struct Thread<'scope, I, R> {
    /// Thread handle object.
    handle: ScopedJoinHandle<'scope, ()>,
    /// Storage for this thread's computation output.
    output: Arc<Mutex<Option<VoteAccumulator<I, R>>>>,
}

impl<'scope, I, R> ThreadPool<'scope, I, R>
where
    I: Integer + Send + Sync + 'scope,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I> + Send + Sync + 'scope,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    /// Creates a new pool tied to the given scope, with the given number of
    /// threads and references to the necessary election inputs.
    pub fn new<'e>(
        thread_scope: &'scope Scope<'scope, 'e>,
        num_threads: NonZeroUsize,
        election: &'e Election,
        pascal: Option<&'e [Vec<I>]>,
    ) -> Self {
        let color = RoundColor::Blue;
        let num_active_threads = Arc::new(AtomicUsize::new(0));
        let worker_status = Arc::new((Mutex::new(WorkerStatus::Round(color)), Condvar::new()));
        let main_status = Arc::new((Mutex::new(MainStatus::Waiting), Condvar::new()));
        let keep_factors = Arc::new(RwLock::new(Vec::new()));
        let num_ballots = election.ballots.len();
        let threads: Vec<_> = (0..num_threads.into())
            .map(|id| {
                let start = (id * num_ballots) / num_threads;
                let end = ((id + 1) * num_ballots) / num_threads;
                let output = Arc::new(Mutex::new(None));
                let context = ThreadContext {
                    id,
                    num_active_threads: num_active_threads.clone(),
                    worker_status: worker_status.clone(),
                    main_status: main_status.clone(),
                    election,
                    pascal,
                    keep_factors: keep_factors.clone(),
                    range: start..end,
                    output: output.clone(),
                };
                Thread {
                    handle: thread_scope.spawn(move || context.run()),
                    output,
                }
            })
            .collect();
        debug!("[main thread] Spawned threads");

        ThreadPool {
            threads,
            num_active_threads,
            round: Cell::new(color),
            worker_status,
            main_status,
            keep_factors,
        }
    }

    /// Accumulates votes from the election ballots based on the given keep
    /// factors.
    pub fn accumulate_votes(&self, keep_factors: &[R]) -> VoteAccumulator<I, R> {
        {
            let mut keep_factors_guard = self.keep_factors.write().unwrap();
            keep_factors_guard.clear();
            keep_factors_guard.extend_from_slice(keep_factors);
        }

        let num_threads = self.threads.len();
        self.num_active_threads.store(num_threads, Ordering::SeqCst);

        let mut round = self.round.get();
        round.toggle();
        self.round.set(round);

        debug!("[main thread, round {round:?}] Ready to accumulate votes.");

        let (lock, condvar) = &*self.worker_status;
        *lock.lock().unwrap() = WorkerStatus::Round(round);
        condvar.notify_all();

        debug!(
            "[main thread, round {round:?}] Waiting for all threads to finish accumulating votes."
        );

        let (lock, condvar) = &*self.main_status;
        let mut guard = condvar
            .wait_while(lock.lock().unwrap(), |status| {
                *status == MainStatus::Waiting
            })
            .unwrap();
        *guard = MainStatus::Waiting;

        debug!("[main thread, round {round:?}] All threads have now finished accumulating votes.");

        self.threads
            .iter()
            .map(|t| -> VoteAccumulator<I, R> { t.output.lock().unwrap().take().unwrap() })
            .reduce(|a, b| a.reduce(b))
            .unwrap()
    }

    /// Joins all the threads in the pool.
    pub fn join(self) {
        debug!("[main thread] Notifying threads to finish...");
        let (lock, condvar) = &*self.worker_status;
        *lock.lock().unwrap() = WorkerStatus::Finished;
        condvar.notify_all();

        debug!("[main thread] Joining threads in the pool...");
        for (i, t) in self.threads.into_iter().enumerate() {
            let result = t.handle.join();
            debug!("[main thread] Thread {i} joined with result: {result:?}");
        }
        debug!("[main thread] Joined threads.");
    }
}

/// Context object owned by a worker thread.
struct ThreadContext<'e, I, R> {
    /// Thread index.
    id: usize,
    /// Number of worker threads active in the current round.
    num_active_threads: Arc<AtomicUsize>,
    /// Status of the worker threads.
    worker_status: Arc<(Mutex<WorkerStatus>, Condvar)>,
    /// Status of the main thread.
    main_status: Arc<(Mutex<MainStatus>, Condvar)>,
    /// Election input.
    election: &'e Election,
    /// Pre-computed Pascal triangle.
    pascal: Option<&'e [Vec<I>]>,
    /// Keep factors used in the current round.
    keep_factors: Arc<RwLock<Vec<R>>>,
    /// Range of ballots that this worker thread needs to count.
    range: Range<usize>,
    /// Storage for the votes accumulated by this thread.
    output: Arc<Mutex<Option<VoteAccumulator<I, R>>>>,
}

impl<I, R> ThreadContext<'_, I, R>
where
    I: Integer,
    for<'a> &'a I: IntegerRef<I>,
    R: Rational<I>,
    for<'a> &'a R: RationalRef<&'a I, R>,
{
    /// Main function run by this thread.
    fn run(&self) {
        let mut round = RoundColor::Blue;
        loop {
            round.toggle();
            debug!(
                "[thread {}, round {round:?}] Waiting for start signal",
                self.id
            );

            let (lock, condvar) = &*self.worker_status;
            let worker_status: WorkerStatus = *condvar
                .wait_while(lock.lock().unwrap(), |status| match status {
                    WorkerStatus::Finished => false,
                    WorkerStatus::Round(r) => *r != round,
                })
                .unwrap();
            match worker_status {
                WorkerStatus::Finished => {
                    debug!(
                        "[thread {}, round {round:?}] Received finish signal",
                        self.id
                    );
                    break;
                }
                WorkerStatus::Round(r) => {
                    assert_eq!(round, r);
                    debug!(
                        "[thread {}, round {round:?}] Received start signal. Processing...",
                        self.id
                    );

                    self.count_votes();

                    let thread_count = self.num_active_threads.fetch_sub(1, Ordering::SeqCst);
                    assert!(thread_count > 0);
                    debug!(
                        "[thread {}, round {round:?}] Decremented the counter: {}.",
                        self.id,
                        thread_count - 1
                    );
                    if thread_count == 1 {
                        // We're the last thread.
                        debug!(
                            "[thread {}, round {round:?}] We're the last thread. Notifying the main thread.",
                            self.id
                        );

                        let (lock, condvar) = &*self.main_status;
                        *lock.lock().unwrap() = MainStatus::Ready;
                        condvar.notify_one();

                        debug!(
                            "[thread {}, round {round:?}] Notified the main thread.",
                            self.id
                        );
                    } else {
                        debug!(
                            "[thread {}, round {round:?}] Waiting for other threads to finish.",
                            self.id
                        );
                    }
                }
            }
        }
    }

    /// Computes a vote counting round.
    fn count_votes(&self) {
        let mut guard = self.output.lock().unwrap();
        let vote_accumulator: &mut VoteAccumulator<I, R> = guard
            .deref_mut()
            .insert(VoteAccumulator::new(self.election.num_candidates));
        let keep_factors = self.keep_factors.read().unwrap();

        for i in self.range.clone() {
            let ballot = &self.election.ballots[i];
            VoteCount::<I, R>::process_ballot(
                vote_accumulator,
                &keep_factors,
                self.pascal,
                i,
                ballot,
            );
        }
    }
}
