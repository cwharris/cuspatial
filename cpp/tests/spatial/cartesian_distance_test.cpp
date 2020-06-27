/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <vector>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cuspatial/cartesian_distance.hpp>
#include <cuspatial/detail/cartesian_product.cuh>
#include <cuspatial/error.hpp>
#include "gtest/gtest.h"
#include "thrust/binary_search.h"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

using namespace cudf;
using namespace test;

template <typename T>
struct CartesianProductTest : public BaseFixture {
};

using TestTypes = Types<double>;

TYPED_TEST_CASE(CartesianProductTest, TestTypes);

TYPED_TEST(CartesianProductTest, Traversal)
{
  auto group_a_offsets_end = 6;
  auto group_a_offsets     = std::vector<int32_t>{0, 3, 4};
  auto group_b_offsets_end = 6;
  auto group_b_offsets     = std::vector<int32_t>{0, 2, 5};

  //     A    A    B    B    B    C
  //   +----+----+----+----+----+----+
  // D : 00   10 : 20   30   40 : 50 :
  //   +    +    +    +    +    +    +
  // D : 01   11 : 21   31   41 : 51 :
  //   +    +    +    +    +    +    +
  // D : 02   12 : 22   32   42 : 52 :
  //   +----+----+----+----+----+----+
  // E : 03   13 : 23   33   43 : 53 :
  //   +----+----+----+----+----+----+
  // F : 04   14 : 24   34   44 : 54 :
  //   +    +    +    +    +    +    +
  // F : 05   15 : 25   35   45 : 55 :
  //   +----+----+----+----+----+----+

  auto gcp_iter =
    cuspatial::detail::make_grouped_cartesian_product_iterator(group_a_offsets_end,
                                                               group_b_offsets_end,
                                                               group_a_offsets.size(),
                                                               group_b_offsets.size(),
                                                               group_a_offsets.cbegin(),
                                                               group_b_offsets.cbegin());

  auto expected_element_pairs = std::vector<std::pair<int32_t, int32_t>>{
    {0, 0}, {0, 1}, {0, 2},  //
    {1, 0}, {1, 1}, {1, 2},  //

    {2, 0}, {2, 1}, {2, 2},  //
    {3, 0}, {3, 1}, {3, 2},  //
    {4, 0}, {4, 1}, {4, 2},  //

    {5, 0}, {5, 1}, {5, 2},  //

    {0, 3},  //
    {1, 3},  //

    {2, 3},  //
    {3, 3},  //
    {4, 3},  //

    {5, 3},  //

    {0, 4}, {0, 5},  //
    {1, 4}, {1, 5},  //

    {2, 4}, {2, 5},  //
    {3, 4}, {3, 5},  //
    {4, 4}, {4, 5},  //

    {5, 4}, {5, 5},  //
  };

  auto expected_group_pairs = std::vector<std::pair<int32_t, int32_t>>{
    {0, 0}, {0, 0}, {0, 0},  //
    {0, 0}, {0, 0}, {0, 0},  //

    {0, 1}, {0, 1}, {0, 1},  //
    {0, 1}, {0, 1}, {0, 1},  //
    {0, 1}, {0, 1}, {0, 1},  //

    {0, 2}, {0, 2}, {0, 2},  //

    {1, 0},  //
    {1, 0},  //

    {1, 1},  //
    {1, 1},  //
    {1, 1},  //

    {1, 2},  //

    {2, 0}, {2, 0},  //
    {2, 0}, {2, 0},  //

    {2, 1}, {2, 1},  //
    {2, 1}, {2, 1},  //
    {2, 1}, {2, 1},  //

    {2, 2}, {2, 2},  //
  };

  auto expected_group_pair_begins = std::vector<int32_t>{0,  0,  0,  0,  0,  0,               //
                                                         12, 12, 12, 12, 12, 12, 12, 12, 12,  //
                                                         30, 30, 30,                          //
                                                         6,  6,                               //
                                                         8,  8,  8,  8,                       //
                                                         21, 21, 21,                          //
                                                         24, 24, 24, 24, 24, 24,              //
                                                         33,                                  //
                                                         34, 34};

  auto num_cartesian = group_a_offsets_end * group_b_offsets_end;

  for (auto i = 0; i < num_cartesian; i++) {
    auto expected_group_pair = expected_group_pairs[i];
    auto actual              = *(gcp_iter + i);

    EXPECT_EQ(expected_group_pair.first, actual.group_a.idx);
    EXPECT_EQ(expected_group_pair.second, actual.group_b.idx);
  }
}

// TYPED_TEST(HausdorffTest, Empty)
// {
//   using T = TypeParam;

//   auto x             = cudf::test::fixed_width_column_wrapper<T>({});
//   auto y             = cudf::test::fixed_width_column_wrapper<T>({});
//   auto shape_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({});

//   auto expected = cudf::test::fixed_width_column_wrapper<T>({});

//   auto actual = cuspatial::cartesian_distance(x, y, shape_offsets);

//   expect_columns_equivalent(expected, actual->view(), true);
// }

// TYPED_TEST(HausdorffTest, TwoTriangles)
// {
//   using T = TypeParam;

//   auto x             = cudf::test::fixed_width_column_wrapper<T>({-3, -1, -1});
//   auto y             = cudf::test::fixed_width_column_wrapper<T>({0, 1, -1});
//   auto shape_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0});

//   auto expected = cudf::test::fixed_width_column_wrapper<T>({0});

//   auto actual = cuspatial::cartesian_distance(x, y, shape_offsets);

//   expect_columns_equivalent(expected, actual->view(), true);
// }
