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

        fn into_iter(self) -> impl Iterator<Item = LogRecord> {
            LOGS.take().into_iter()
        }

        #[track_caller]
        pub fn check_logs<'a>(self, expected: impl IntoIterator<Item = (Level, &'a str, &'a str)>) {
            let report = self
                .into_iter()
                .map(|record| (record.level, record.target, record.message))
                .collect::<Vec<_>>();

            let expected_report = expected
                .into_iter()
                .map(|(level, target, msg)| (level, target.to_owned(), msg.to_owned()))
                .collect::<Vec<_>>();

            assert_eq!(report, expected_report);
        }

        #[track_caller]
        pub fn check_target_logs<'a>(
            self,
            target: &str,
            expected: impl IntoIterator<Item = (Level, &'a str)>,
        ) {
            self.check_logs(
                expected
                    .into_iter()
                    .map(|(level, msg)| (level, target, msg)),
            );
        }

        #[track_caller]
        pub fn check_target_level_logs(self, target: &str, level: Level, expected: &str) {
            let mut report = String::new();
            for record in self.into_iter() {
                assert_eq!(record.target, target);
                assert_eq!(record.level, level);
                report.push_str(&record.message);
                report.push('\n');
            }

            assert_eq!(report, expected);
        }

        #[track_caller]
        pub fn check_logs_at_target_level(self, target: &str, level: Level, expected: &str) {
            let mut report = String::new();
            for record in self.into_iter() {
                if record.level == level && record.target == target {
                    report.push_str(&record.message);
                    report.push('\n');
                }
            }

            assert_eq!(report, expected);
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
