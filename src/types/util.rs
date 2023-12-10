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

//! Utilities for election types.

use std::collections::BTreeMap;

pub fn count_vec_allocations<T>(allocations: &mut BTreeMap<usize, usize>, v: &Vec<T>) {
    let size = v.capacity() * std::mem::size_of::<T>();
    *allocations.entry(size).or_insert(0) += 1;
}

pub fn count_slice_allocations<T>(allocations: &mut BTreeMap<usize, usize>, v: &[T]) {
    let size = std::mem::size_of_val(v);
    // Heuristic: allocator aligns to 32 bytes.
    let size = (size + 31) & !31;
    *allocations.entry(size).or_insert(0) += 1;
}

/// Returns the address of the beginning of the slice, or none if the slice is
/// empty.
pub fn address_of_slice<T>(v: &[T]) -> Option<usize> {
    if v.is_empty() {
        None
    } else {
        Some(v.as_ptr() as usize)
    }
}
