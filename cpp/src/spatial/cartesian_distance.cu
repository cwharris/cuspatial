#include "rmm/thrust_rmm_allocator.h"
#include "utility/size_from_offsets.cuh"

#include <cuspatial/detail/cartesian_product.cuh>
#include <cuspatial/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace cuspatial {

template <typename T>
__device__ T directed_distance_line_segment_to_point(
  T const ax, T const ay, T const bx, T const by, T const px, T const py)
{
  // difference
  auto dx = bx - ax;
  auto dy = by - ay;

  if (ax == bx and ay == by) { return std::numeric_limits<T>::quiet_NaN(); }

  auto const segment_length = hypot(bx - ax, by - ay);

  // tangent
  auto const tx = dx / segment_length;
  auto const ty = dy / segment_length;

  auto segment_travel = (px - ax) * tx + (py - ax) * ty;

  if (segment_travel > segment_length) { return hypot(bx - px, by - py); }
  if (segment_travel < 0) { return hypot(ax - px, ay - py); }

  // normal
  auto const nx = -ty;
  auto const ny = tx;

  auto const segment_offset = px * nx + py * ny;

  return abs(segment_offset);
}

template <typename T>
using edge = thrust::tuple<T, T, T, T>;

template <typename T>
struct distance_functor {
  cudf::column_device_view const xs;
  cudf::column_device_view const ys;

  __device__ T operator()(detail::cartesian_product_group_index const& idx)
  {
    auto a1_idx = idx.group_a.offset + idx.element_a_idx;
    auto a2_idx = idx.group_a.offset + (idx.element_a_idx + 1) % idx.group_a.size;

    auto b1_idx = idx.group_b.offset + idx.element_b_idx;
    auto b2_idx = idx.group_b.offset + (idx.element_b_idx + 1) % idx.group_b.size;

    auto a1_x = xs.element<T>(a1_idx), a1_y = ys.element<T>(a1_idx);
    auto a2_x = xs.element<T>(a2_idx), a2_y = ys.element<T>(a2_idx);
    auto b1_x = xs.element<T>(b1_idx), b1_y = ys.element<T>(b1_idx);
    auto b2_x = xs.element<T>(b2_idx), b2_y = ys.element<T>(b2_idx);

    return 0;
    // auto elements = idx.get_absolute_idx_pair();
    // auto a1       = elements.first;
    // auto b1       = elements.second;
    // auto a2       = idx.wrap_absolute_idx_a(a1 + 1);
    // auto b2       = idx.wrap_absolute_idx_b(b1 + 1);
    // auto a1_x     = xs.element<T>(a1);
    // auto a1_y     = ys.element<T>(a1);
    // auto a2_x     = xs.element<T>(a2);
    // auto a2_y     = ys.element<T>(a2);
    // auto b1_x     = xs.element<T>(b1);
    // auto b1_y     = ys.element<T>(b1);
    // auto b2_x     = xs.element<T>(b2);
    // auto b2_y     = ys.element<T>(b2);

    // return min(min(directed_distance_line_segment_to_point(a1_x, a1_y, a2_x, a2_y, b1_x, b1_y),
    //                directed_distance_line_segment_to_point(a1_x, a1_y, a2_x, a2_y, b2_x, b2_y)),
    //            min(directed_distance_line_segment_to_point(b1_x, b1_y, b2_x, b2_y, a1_x, a1_y),
    //                directed_distance_line_segment_to_point(b1_x, b1_y, b2_x, b2_y, a2_x, a2_y)));
  }
};

struct key_comp_functor {
  bool __device__ operator()(detail::cartesian_product_group_index const& lhs,
                             detail::cartesian_product_group_index const& rhs)
  {
    return lhs.group_a.idx == rhs.group_a.idx &&  //
           lhs.group_b.idx == rhs.group_b.idx;
  }
};

struct cartesian_distance_functor {
  template <typename T,
            std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr,
            typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
  {
    CUSPATIAL_FAIL("unsupported type");
  }

  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& xs,
                                           cudf::column_view const& ys,
                                           cudf::column_view const& shape_offsets,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream)
  {
    auto const num_shapes = shape_offsets.size();

    if (num_shapes == 0) { return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<T>()}); }

    auto num_points = xs.size();

    auto gcp_iter =
      detail::make_grouped_cartesian_product_iterator(num_points,
                                                      num_points,
                                                      num_shapes,
                                                      num_shapes,
                                                      shape_offsets.begin<cudf::size_type>(),
                                                      shape_offsets.begin<cudf::size_type>());

    auto d_xs = cudf::column_device_view::create(xs);
    auto d_ys = cudf::column_device_view::create(ys);

    auto distance_iter =
      thrust::make_transform_iterator(gcp_iter, distance_functor<T>{*d_xs, *d_ys});

    auto result = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<T>()},
                                                num_shapes * num_shapes,
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);

    auto num_cartesian = num_points * num_points;

    thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                          gcp_iter,
                          gcp_iter + num_cartesian,
                          distance_iter,
                          thrust::make_discard_iterator(),
                          result->mutable_view().begin<T>(),
                          key_comp_functor{},
                          thrust::minimum<T>());

    return result;
  }
};

std::unique_ptr<cudf::column> cartesian_distance(
  cudf::column_view const& xs,
  cudf::column_view const& ys,
  cudf::column_view const& shape_offsets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  CUSPATIAL_EXPECTS(xs.size() == ys.size(), "sizes don't match.");
  CUSPATIAL_EXPECTS(xs.size() >= shape_offsets.size(), "points per space is wrong.");

  return cudf::type_dispatcher(
    xs.type(), cartesian_distance_functor{}, xs, ys, shape_offsets, mr, stream);
}

}  // namespace cuspatial