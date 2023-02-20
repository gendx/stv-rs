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

#[cfg(test)]
pub mod log_tester {
    use log::{Level, LevelFilter, Metadata, Record};
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;

    pub struct LogRecord {
        pub level: Level,
        pub target: String,
        pub message: String,
    }

    thread_local! {
        static ACTIVE: Cell<bool> = Cell::new(false);
        static LOGS: RefCell<VecDeque<LogRecord>> = RefCell::new(VecDeque::new());
    }

    pub struct ThreadLocalLogger;

    impl ThreadLocalLogger {
        pub fn start() -> Self {
            // set_logger only succeeds the first time, but the error isn't a problem.
            let _ = log::set_logger(&LoggerImpl);
            log::set_max_level(LevelFilter::Trace);
            let old = ACTIVE.replace(true);
            assert!(!old);
            ThreadLocalLogger
        }

        pub fn into_iter(self) -> impl Iterator<Item = LogRecord> {
            LOGS.take().into_iter()
        }
    }

    impl Drop for ThreadLocalLogger {
        fn drop(&mut self) {
            let old = ACTIVE.replace(false);
            assert!(old);
            LOGS.with_borrow_mut(|logs| logs.clear());
        }
    }

    struct LoggerImpl;

    impl log::Log for LoggerImpl {
        fn enabled(&self, _metadata: &Metadata) -> bool {
            ACTIVE.get()
        }

        fn log(&self, record: &Record) {
            if self.enabled(record.metadata()) {
                LOGS.with_borrow_mut(|logs| {
                    logs.push_back(LogRecord {
                        level: record.level(),
                        target: record.target().to_owned(),
                        message: format!("{}", record.args()),
                    })
                });
            }
        }

        fn flush(&self) {}
    }
}
