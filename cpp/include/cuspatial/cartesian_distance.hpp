#include <cudf/types.hpp>

#include <memory>

namespace cuspatial {

std::unique_ptr<cudf::column> cartesian_distance(
  cudf::column_view const& xs,
  cudf::column_view const& ys,
  cudf::column_view const& shape_offsets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

}
