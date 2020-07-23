/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuspatial/detail/cartesian_product_group_index_iterator.cuh>
#include <cuspatial/error.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <limits>
#include <memory>
#include "thrust/functional.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

using size_type = cudf::size_type;

namespace cuspatial {
namespace detail {
namespace {

template <typename T>
__device__ inline int segment_orientation(T px, T py, T qx, T qy, T rx, T ry)
{
  int val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy);

  if (val == 0) return 0;  // colinear

  return (val > 0) ? 1 : 2;  // clock or counter-clock wise
}

template <typename T>
__device__ inline bool segment_intersect(T p1x, T p1y, T q1x, T q1y, T p2x, T p2y, T q2x, T q2y)
{
  // Find the four segment_orientations needed for general and
  // special cases
  int o1 = segment_orientation(p1x, p1y, q1x, q1y, p2x, p2y);
  int o2 = segment_orientation(p1x, p1y, q1x, q1y, q2x, q2y);
  int o3 = segment_orientation(p2x, p2y, q2x, q2y, p1x, p1y);
  int o4 = segment_orientation(p2x, p2y, q2x, q2y, q1x, q1y);

  // General case
  return o1 != o2 && o3 != o4;

  // // Special Cases
  // // p1, q1 and p2 are colinear and p2 lies on segment p1q1
  // if (o1 == 0 && onSegment(p1x, p1y, p2x, p2y, q1x, q1y)) return true;
  // // p1, q1 and q2 are colinear and q2 lies on segment p1q1
  // if (o2 == 0 && onSegment(p1x, p1y, q2x, q2y, q1x, q1y)) return true;
  // // p2, q2 and p1 are colinear and p1 lies on segment p2q2
  // if (o3 == 0 && onSegment(p2x, p2y, p1x, p1y, q2x, q2y)) return true;
  // // p2, q2 and q1 are colinear and q1 lies on segment p2q2
  // if (o4 == 0 && onSegment(p2x, p2y, q1x, q1y, q2x, q2y)) return true;

  return false;  // Doesn't fall in any of the above cases
}

/**
 * @brief Calculates segment-segment intersection
 *
 * Given a `cartesian_product_group_index` and two columns representing `x` and `y` coordinates,
 * determine if the indexed segment in group `a` intersects with the index segment in group `b`.
 *
 * If group `a` contains two or more points, calculate segment-point distance.
 * If group `a` contains only a single point, calculate point-point distance.
 */
template <typename T>
struct segment_intersect_predicate {
  cudf::column_device_view xs;
  cudf::column_device_view ys;

  bool __device__ operator()(cartesian_product_group_index idx)
  {
    auto const idx_a = idx.group_a.offset + (idx.element_a_idx);
    auto const idx_b = idx.group_a.offset + (idx.element_a_idx + 1) % idx.group_a.size;
    auto const idx_c = idx.group_b.offset + (idx.element_b_idx);
    auto const idx_d = idx.group_b.offset + (idx.element_b_idx + 1) % idx.group_b.size;

    auto const ax = xs.element<T>(idx_a);
    auto const ay = ys.element<T>(idx_a);
    auto const bx = xs.element<T>(idx_b);
    auto const by = ys.element<T>(idx_b);
    auto const cx = xs.element<T>(idx_c);
    auto const cy = ys.element<T>(idx_c);
    auto const dx = xs.element<T>(idx_d);
    auto const dy = ys.element<T>(idx_d);

    return segment_intersect(ax, ay, bx, by, cx, cy, dx, dy);
  }
};

struct polygon_intersection_functor {
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& xs,
    cudf::column_view const& ys,
    cudf::column_view const& space_offsets,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
  {
    size_type num_points  = xs.size();
    size_type num_spaces  = space_offsets.size();
    size_type num_results = num_spaces * num_spaces;

    if (num_results == 0) {
      return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<bool>()});
    }

    // ===== Make Separation and Key Iterators =====================================================

    auto cartesian_iter = make_cartesian_product_group_index_iterator(
      num_points, num_spaces, space_offsets.begin<cudf::size_type>());

    auto cartesian_key_iter = thrust::make_transform_iterator(
      cartesian_iter, [] __device__(cartesian_product_group_index idx) {
        return thrust::make_pair(idx.group_a.idx, idx.group_b.idx);
      });

    auto d_xs = cudf::column_device_view::create(xs);
    auto d_ys = cudf::column_device_view::create(ys);

    auto intersection_iter =
      thrust::make_transform_iterator(cartesian_iter, segment_intersect_predicate<T>{*d_xs, *d_ys});

    // ===== Materialize ===========================================================================

    auto result = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<bool>()},
                                                num_results,
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);

    auto num_cartesian = num_points * num_points;

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          cartesian_key_iter,
                          cartesian_key_iter + num_cartesian,
                          intersection_iter,
                          thrust::make_discard_iterator(),
                          result->mutable_view().begin<bool>(),
                          thrust::equal_to<thrust::pair<int32_t, int32_t>>(),
                          thrust::maximum<bool>());

    return result;
  }
};

}  // namespace
}  // namespace detail

std::unique_ptr<cudf::column> polygon_intersect(cudf::column_view const& xs,
                                                cudf::column_view const& ys,
                                                cudf::column_view const& offsets,
                                                rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(xs.type() == ys.type(), "Inputs `xs` and `ys` must have same type.");
  CUSPATIAL_EXPECTS(xs.size() == ys.size(), "Inputs `xs` and `ys` must have same length.");

  CUSPATIAL_EXPECTS(not xs.has_nulls() and not ys.has_nulls() and not offsets.has_nulls(),
                    "Inputs must not have nulls.");

  CUSPATIAL_EXPECTS(xs.size() >= offsets.size(), "At least one point is required for each space");

  cudaStream_t stream = 0;

  return cudf::type_dispatcher(
    xs.type(), detail::polygon_intersection_functor(), xs, ys, offsets, mr, stream);
}

}  // namespace cuspatial
