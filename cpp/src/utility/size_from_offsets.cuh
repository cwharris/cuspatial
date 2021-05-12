/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cuspatial {
namespace detail {

template <typename OffsetIterator>
struct size_from_offsets_functor {
  uint32_t const num_offsets;
  uint32_t const num_elements;
  OffsetIterator const offsets;

  uint32_t inline __device__ operator()(uint64_t const group_idx)
  {
    auto const group_idx_next = group_idx + 1;
    auto const group_begin    = *(offsets + group_idx);
    auto const group_end =
      group_idx_next >= num_offsets ? num_elements : *(offsets + group_idx_next);

    return group_end - group_begin;
  }
};

template<typename OffsetIterator>
auto make_size_from_offset_iterator(uint32_t num_offsets,
                                    uint32_t num_elements,
                                    OffsetIterator const offsets)
{
  return thrust::make_transform_iterator(  //
    thrust::make_counting_iterator(uint64_t{0}),
    detail::size_from_offsets_functor<OffsetIterator>{
      num_offsets, num_elements, offsets});
}

}  // namespace detail
}  // namespace cuspatial
