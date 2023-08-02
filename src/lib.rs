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

//! Single Transferable Vote implementation in Rust. The Meek algorithm is
//! implemented in the corresponding module.

#![forbid(missing_docs, unsafe_code)]
#![cfg_attr(test, feature(test, local_key_cell_methods))]

#[cfg(test)]
extern crate test;

pub mod arithmetic;
pub mod blt;
pub mod meek;
pub mod parse;
pub mod pbv;
pub mod types;
#[cfg(test)]
mod util;
mod vote_count;
