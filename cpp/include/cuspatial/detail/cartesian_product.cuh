/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required point_b_y applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <utility>
#include "utility/size_from_offsets.cuh"

namespace cuspatial {

namespace detail {

struct cartesian_product_group {
  int32_t idx;
  int32_t size;
  int32_t offset;
};

struct cartesian_product_group_index {
  cartesian_product_group group_a;
  cartesian_product_group group_b;
  int32_t element_a_idx;
  int32_t element_b_idx;
};

template <typename OffsetIteratorLHS,
          typename OffsetIteratorRHS,
          typename SizeIteratorLHS,
          typename SizeIteratorRHS,
          typename LookupIteratorLHS,
          typename LookupIteratorRHS>
struct grouped_cartesian_product_traversal_functor {
  int32_t const num_elements_a;
  int32_t const num_elements_b;
  int32_t const num_groups_a;
  int32_t const num_groups_b;
  SizeIteratorLHS const group_sizes_a;
  SizeIteratorRHS const group_sizes_b;
  OffsetIteratorLHS const group_offsets_a;
  OffsetIteratorRHS const group_offsets_b;
  LookupIteratorLHS const group_lookup_a;
  LookupIteratorRHS const group_lookup_b;

  __device__ cartesian_product_group_index operator()(int64_t const idx) const
  {
    // first dimension
    int32_t group_a_idx       = *(group_lookup_a + idx / num_elements_b);
    auto group_a_size         = *(group_sizes_a + group_a_idx);
    auto group_a_offset       = *(group_offsets_a + group_a_idx);
    auto group_a_block_offset = group_a_offset * num_elements_b;

    // second dimension
    int32_t group_b_idx       = *(group_lookup_b + (idx - group_a_block_offset) / group_a_size);
    auto group_b_size         = *(group_sizes_b + group_b_idx);
    auto group_b_offset       = *(group_offsets_b + group_b_idx);
    auto group_b_block_offset = group_b_offset * group_a_size;

    // relative index
    auto relative_idx           = idx - group_a_block_offset - group_b_block_offset;
    int32_t group_a_element_idx = relative_idx % group_a_size;
    int32_t group_b_element_idx = relative_idx / group_a_size;

    return cartesian_product_group_index{{group_a_idx, group_a_size, group_a_offset},
                                         {group_b_idx, group_b_size, group_b_offset},
                                         group_a_element_idx,
                                         group_b_element_idx};
  }
};

template <typename OffsetIterator>
struct group_lookup_functor {
  OffsetIterator group_offsets;
  int32_t group_count;
  __device__ int32_t operator()(int32_t element_idx) const
  {
    return thrust::upper_bound(
             thrust::seq, group_offsets, group_offsets + group_count, element_idx) -
           group_offsets - 1;
  }
};

template <typename OffsetIteratorLHS, typename OffsetIteratorRHS>
auto make_grouped_cartesian_product_iterator(int32_t const num_elements_a,
                                             int32_t const num_elements_b,
                                             int32_t const num_groups_a,
                                             int32_t const num_groups_b,
                                             OffsetIteratorLHS const group_offsets_a,
                                             OffsetIteratorRHS const group_offsets_b)
{
  auto count_iter = thrust::make_counting_iterator(0);

  using SizeFunctorLHS = detail::size_from_offsets_functor<OffsetIteratorLHS>;
  using SizeFunctorRHS = detail::size_from_offsets_functor<OffsetIteratorLHS>;

  auto group_sizes_a = thrust::make_transform_iterator(
    count_iter, SizeFunctorLHS{group_offsets_a, num_groups_a, num_elements_a});
  auto group_sizes_b = thrust::make_transform_iterator(
    count_iter, SizeFunctorRHS{group_offsets_b, num_groups_b, num_elements_b});
  auto group_lookup_a = thrust::make_transform_iterator(
    count_iter, group_lookup_functor<OffsetIteratorLHS>{group_offsets_a, num_groups_a});
  auto group_lookup_b = thrust::make_transform_iterator(
    count_iter, group_lookup_functor<OffsetIteratorLHS>{group_offsets_b, num_groups_b});

  using SizeIteratorLHS   = decltype(group_sizes_a);
  using SizeIteratorRHS   = decltype(group_sizes_b);
  using LookupIteratorLHS = decltype(group_lookup_a);
  using LookupIteratorRHS = decltype(group_lookup_b);

  using TraversalFunctor = grouped_cartesian_product_traversal_functor<OffsetIteratorLHS,
                                                                       OffsetIteratorRHS,
                                                                       SizeIteratorLHS,
                                                                       SizeIteratorRHS,
                                                                       LookupIteratorLHS,
                                                                       LookupIteratorRHS>;

  auto traversal = TraversalFunctor{num_elements_a,
                                    num_elements_b,
                                    num_groups_a,
                                    num_groups_b,
                                    group_sizes_a,
                                    group_sizes_b,
                                    group_offsets_a,
                                    group_offsets_b,
                                    group_lookup_a,
                                    group_lookup_b};

  return thrust::make_transform_iterator(count_iter, traversal);
}

template <typename OffsetIterator>
auto make_grouped_cartesian_product_iterator(int32_t const element_count,
                                             int32_t const group_count,
                                             OffsetIterator const group_offsets)
{
  return make_grouped_cartesian_product_iterator(
    element_count, element_count, group_count, group_count, group_offsets, group_offsets);
}

}  // namespace detail

}  // namespace cuspatial