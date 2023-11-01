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
use crate::parallelism::range::{
    FixedRangeFactory, Range, RangeFactory, RangeOrchestrator, WorkStealingRangeFactory,
};
use crate::types::{BallotCursor, Election};
use crate::vote_count::{VoteAccumulator, VoteCount};
use log::{debug, error, warn};
// Platforms that support `libc::sched_setaffinity()`.
#[cfg(any(
    target_os = "android",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "linux"
))]
use nix::{
    sched::{sched_setaffinity, CpuSet},
    unistd::Pid,
};
use std::cell::Cell;
use std::num::NonZeroUsize;
use std::ops::DerefMut;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock};
use std::thread::{Scope, ScopedJoinHandle};

/// Status of the main thread.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MainStatus {
    /// The main thread is waiting for the worker threads to finish a round.
    Waiting,
    /// The main thread is ready to prepare the next round.
    Ready,
    /// One of the worker threads panicked.
    WorkerPanic,
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

/// An ergonomic wrapper around a [`Mutex`]-[`Condvar`] pair.
struct Status<T> {
    mutex: Mutex<T>,
    condvar: Condvar,
}

impl<T> Status<T> {
    /// Creates a new status initialized with the given value.
    fn new(t: T) -> Self {
        Self {
            mutex: Mutex::new(t),
            condvar: Condvar::new(),
        }
    }

    /// Attempts to set the status to the given value and notifies one waiting
    /// thread.
    ///
    /// Fails if the [`Mutex`] is poisoned.
    fn try_notify_one(&self, t: T) -> Result<(), PoisonError<MutexGuard<'_, T>>> {
        *self.mutex.lock()? = t;
        self.condvar.notify_one();
        Ok(())
    }

    /// If the predicate is true on this status, sets the status to the given
    /// value and notifies one waiting thread.
    fn notify_one_if(&self, predicate: impl Fn(&T) -> bool, t: T) {
        let mut locked = self.mutex.lock().unwrap();
        if predicate(&*locked) {
            *locked = t;
            self.condvar.notify_one();
        }
    }

    /// Sets the status to the given value and notifies all waiting threads.
    fn notify_all(&self, t: T) {
        *self.mutex.lock().unwrap() = t;
        self.condvar.notify_all();
    }

    /// Waits until the predicate is true on this status.
    ///
    /// This returns a [`MutexGuard`], allowing to further inspect or modify the
    /// status.
    fn wait_while(&self, predicate: impl FnMut(&mut T) -> bool) -> MutexGuard<T> {
        self.condvar
            .wait_while(self.mutex.lock().unwrap(), predicate)
            .unwrap()
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
    worker_status: Arc<Status<WorkerStatus>>,
    /// Status of the main thread.
    main_status: Arc<Status<MainStatus>>,
    /// Storage for the keep factors, used as input of the current round by the
    /// worker threads.
    keep_factors: Arc<RwLock<Vec<R>>>,
    /// Orchestrator for the work ranges distributed to the threads. This is a
    /// dynamic object to avoid making the range type a parameter of
    /// everything.
    range_orchestrator: Box<dyn RangeOrchestrator>,
}

/// Handle to a thread in the pool.
struct Thread<'scope, I, R> {
    /// Thread handle object.
    handle: ScopedJoinHandle<'scope, ()>,
    /// Storage for this thread's computation output.
    output: Arc<Mutex<Option<VoteAccumulator<I, R>>>>,
}

/// Strategy to distribute ranges of work items among threads.
pub enum RangeStrategy {
    /// Each thread processes a fixed range of items.
    Fixed,
    /// Threads can steal work from each other.
    WorkStealing,
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
        range_strategy: RangeStrategy,
        election: &'e Election,
        pascal: Option<&'e [Vec<I>]>,
    ) -> Self {
        let num_threads: usize = num_threads.into();
        let num_ballots = election.num_ballots();
        match range_strategy {
            RangeStrategy::Fixed => Self::new_with_factory(
                thread_scope,
                num_threads,
                FixedRangeFactory::new(num_ballots, num_threads),
                election,
                pascal,
            ),
            RangeStrategy::WorkStealing => Self::new_with_factory(
                thread_scope,
                num_threads,
                WorkStealingRangeFactory::new(num_ballots, num_threads),
                election,
                pascal,
            ),
        }
    }

    fn new_with_factory<'e, RnFactory: RangeFactory>(
        thread_scope: &'scope Scope<'scope, 'e>,
        num_threads: usize,
        range_factory: RnFactory,
        election: &'e Election,
        pascal: Option<&'e [Vec<I>]>,
    ) -> Self
    where
        RnFactory::Rn: 'scope + Send,
        RnFactory::Orchestrator: 'static,
    {
        let color = RoundColor::Blue;
        let num_active_threads = Arc::new(AtomicUsize::new(0));
        let worker_status = Arc::new(Status::new(WorkerStatus::Round(color)));
        let main_status = Arc::new(Status::new(MainStatus::Waiting));
        let keep_factors = Arc::new(RwLock::new(Vec::new()));

        #[cfg(not(any(
            target_os = "android",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "linux"
        )))]
        warn!("Pinning threads to CPUs is not implemented on this platform.");
        let threads = (0..num_threads)
            .map(|id| {
                let output = Arc::new(Mutex::new(None));
                let context = ThreadContext {
                    id,
                    num_active_threads: num_active_threads.clone(),
                    worker_status: worker_status.clone(),
                    main_status: main_status.clone(),
                    election,
                    pascal,
                    keep_factors: keep_factors.clone(),
                    range: range_factory.range(id),
                    output: output.clone(),
                };
                Thread {
                    handle: thread_scope.spawn(move || {
                        #[cfg(any(
                            target_os = "android",
                            target_os = "dragonfly",
                            target_os = "freebsd",
                            target_os = "linux"
                        ))]
                        {
                            let mut cpu_set = CpuSet::new();
                            if let Err(e) = cpu_set.set(id) {
                                warn!("Failed to set CPU affinity for thread #{id}: {e}");
                            } else if let Err(e) = sched_setaffinity(Pid::from_raw(0), &cpu_set) {
                                warn!("Failed to set CPU affinity for thread #{id}: {e}");
                            } else {
                                debug!("Pinned thread #{id} to CPU #{id}");
                            }
                        }
                        context.run()
                    }),
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
            range_orchestrator: Box::new(range_factory.orchestrator()),
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
        self.range_orchestrator.reset_ranges();

        let num_threads = self.threads.len();
        self.num_active_threads.store(num_threads, Ordering::SeqCst);

        let mut round = self.round.get();
        round.toggle();
        self.round.set(round);

        debug!("[main thread, round {round:?}] Ready to accumulate votes.");

        self.worker_status.notify_all(WorkerStatus::Round(round));

        debug!(
            "[main thread, round {round:?}] Waiting for all threads to finish accumulating votes."
        );

        let mut guard = self
            .main_status
            .wait_while(|status| *status == MainStatus::Waiting);
        if *guard == MainStatus::WorkerPanic {
            error!("[main thread] A worker thread panicked!");
            panic!("A worker thread panicked!");
        }
        *guard = MainStatus::Waiting;
        drop(guard);

        debug!("[main thread, round {round:?}] All threads have now finished accumulating votes.");

        self.threads
            .iter()
            .map(|t| -> VoteAccumulator<I, R> { t.output.lock().unwrap().take().unwrap() })
            .reduce(|a, b| a.reduce(b))
            .unwrap()
    }
}

impl<I, R> Drop for ThreadPool<'_, I, R> {
    /// Joins all the threads in the pool.
    fn drop(&mut self) {
        debug!("[main thread] Notifying threads to finish...");
        self.worker_status.notify_all(WorkerStatus::Finished);

        debug!("[main thread] Joining threads in the pool...");
        for (i, t) in self.threads.drain(..).enumerate() {
            let result = t.handle.join();
            match result {
                Ok(_) => debug!("[main thread] Thread {i} joined with result: {result:?}"),
                Err(_) => error!("[main thread] Thread {i} joined with result: {result:?}"),
            }
        }
        debug!("[main thread] Joined threads.");

        #[cfg(feature = "log_parallelism")]
        self.range_orchestrator.print_statistics();
    }
}

/// Context object owned by a worker thread.
struct ThreadContext<'e, I, R, Rn: Range> {
    /// Thread index.
    id: usize,
    /// Number of worker threads active in the current round.
    num_active_threads: Arc<AtomicUsize>,
    /// Status of the worker threads.
    worker_status: Arc<Status<WorkerStatus>>,
    /// Status of the main thread.
    main_status: Arc<Status<MainStatus>>,
    /// Election input.
    election: &'e Election,
    /// Pre-computed Pascal triangle.
    pascal: Option<&'e [Vec<I>]>,
    /// Keep factors used in the current round.
    keep_factors: Arc<RwLock<Vec<R>>>,
    /// Range of ballots that this worker thread needs to count.
    range: Rn,
    /// Storage for the votes accumulated by this thread.
    output: Arc<Mutex<Option<VoteAccumulator<I, R>>>>,
}

impl<I, R, Rn: Range> ThreadContext<'_, I, R, Rn>
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

            let worker_status: WorkerStatus =
                *self.worker_status.wait_while(|status| match status {
                    WorkerStatus::Finished => false,
                    WorkerStatus::Round(r) => *r != round,
                });
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

                    // Counting votes may panic, and we want to notify the main thread in that case
                    // to avoid a deadlock.
                    let panic_notifier = PanicNotifier {
                        id: self.id,
                        main_status: &self.main_status,
                    };
                    self.count_votes();
                    std::mem::forget(panic_notifier);

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

                        self.main_status.notify_one_if(
                            |&status| status == MainStatus::Waiting,
                            MainStatus::Ready,
                        );

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

        let ballots = self.election.ballots();
        for i in self.range.iter() {
            let ballot = ballots.at(i).unwrap();
            VoteCount::<I, R>::process_ballot(
                vote_accumulator,
                &keep_factors,
                self.pascal,
                i,
                &ballot,
            );
        }
    }
}

/// Object whose destructor notifies the main thread that a panic happened.
///
/// The way to use this is to create an instance before a section that may
/// panic, and to [`std::mem::forget()`] it at the end of the section. That way:
/// - If a panic happens, the [`std::mem::forget()`] call will be skipped but
///   the destructor will run due to RAII.
/// - If no panic happens, the destructor won't run because this object will be
///   forgotten.
struct PanicNotifier<'a> {
    /// Thread index.
    id: usize,
    /// Status of the main thread.
    main_status: &'a Status<MainStatus>,
}

impl Drop for PanicNotifier<'_> {
    fn drop(&mut self) {
        error!(
            "[thread {}] Detected panic in this thread, notifying the main thread",
            self.id
        );
        if let Err(e) = self.main_status.try_notify_one(MainStatus::WorkerPanic) {
            error!(
                "[thread {}] Failed to notify the main thread, the mutex was poisoned: {e:?}",
                self.id
            );
        }
    }
}
