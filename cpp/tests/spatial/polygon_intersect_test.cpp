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

#include <cuspatial/error.hpp>
#include <cuspatial/polygon_intersect.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <gtest/gtest.h>

#include <vector>
using namespace cudf;
using namespace test;

using TestTypes            = Types<float, double>;
using UnsupportedTestTypes = RemoveIf<ContainedIn<TestTypes>, AllTypes>;

template <typename T>
struct PolygonIntersectTest : public BaseFixture {
};

TYPED_TEST_CASE(PolygonIntersectTest, TestTypes);

TYPED_TEST(PolygonIntersectTest, EmptyInput)
{
  using T = TypeParam;

  auto x             = fixed_width_column_wrapper<T>({});
  auto y             = fixed_width_column_wrapper<T>({});
  auto space_offsets = fixed_width_column_wrapper<size_type>({});

  auto expected = fixed_width_column_wrapper<bool>({});
  auto actual   = cuspatial::polygon_intersect(x, y, space_offsets);

  expect_columns_equivalent(expected, actual->view(), true);
}

TYPED_TEST(PolygonIntersectTest, TwoEdgeX)
{
  using T = TypeParam;

  auto x             = fixed_width_column_wrapper<T>({0, 1, 0, 1});
  auto y             = fixed_width_column_wrapper<T>({0, 1, 1, 0});
  auto space_offsets = fixed_width_column_wrapper<size_type>({0, 2});

  auto expected = fixed_width_column_wrapper<bool>({false, true, true, false});
  auto actual   = cuspatial::polygon_intersect(x, y, space_offsets);

  expect_columns_equal(expected, actual->view(), true);
}

// TYPED_TEST(PolygonIntersectTest, OneShapeSinglePoint)
// {
//   using T = TypeParam;

//   auto x             = fixed_width_column_wrapper<T>({2});
//   auto y             = fixed_width_column_wrapper<T>({2});
//   auto space_offsets = fixed_width_column_wrapper<size_type>({0});

//   auto expected = fixed_width_column_wrapper<bool>({0.0});
//   auto actual   = cuspatial::polygon_intersect(x, y, space_offsets);

//   expect_columns_equivalent(expected, actual->view(), true);
// }

// TYPED_TEST(PolygonIntersectTest, TwoShapesPointToPoint)
// {
//   using T = TypeParam;

//   auto x             = fixed_width_column_wrapper<T>({-1, -2, -3, 1, 2, 3});
//   auto y             = fixed_width_column_wrapper<T>({-1, -3, -2, 1, 3, 2});
//   auto space_offsets = fixed_width_column_wrapper<size_type>({0, 3});

//   auto expected = fixed_width_column_wrapper<bool>({0.0, 2.8284271247461903, 2.8284271247461903,
//   0.0}); auto actual   = cuspatial::polygon_intersect(x, y, space_offsets);

//   expect_columns_equivalent(expected, actual->view(), true);
// }

// TYPED_TEST(PolygonIntersectTest, ZeroLengthEdge)
// {
//   using T = TypeParam;

//   auto x             = fixed_width_column_wrapper<T>({0, 0});
//   auto y             = fixed_width_column_wrapper<T>({0, 0});
//   auto space_offsets = fixed_width_column_wrapper<size_type>({0});

//   auto expected = fixed_width_column_wrapper<bool>({0});
//   auto actual   = cuspatial::polygon_intersect(x, y, space_offsets);

//   expect_columns_equal(expected, actual->view(), true);
// }

// TYPED_TEST(PolygonIntersectTest, MismatchedTypeTest)
// {
//   auto x             = fixed_width_column_wrapper<double>({0, 0});
//   auto y             = fixed_width_column_wrapper<float>({0, 0});
//   auto space_offsets = fixed_width_column_wrapper<size_type>({0});

//   EXPECT_THROW(cuspatial::polygon_intersect(x, y, space_offsets), cuspatial::logic_error);
// }

template <typename T>
struct DirectedPolygonDistanceUnsupportedTypesTest : public BaseFixture {
};

TYPED_TEST_CASE(DirectedPolygonDistanceUnsupportedTypesTest, UnsupportedTestTypes);

TYPED_TEST(DirectedPolygonDistanceUnsupportedTypesTest, InvalidTypeTest)
{
  using T = TypeParam;

  auto x             = fixed_width_column_wrapper<T>({0, 0});
  auto y             = fixed_width_column_wrapper<T>({0, 0});
  auto space_offsets = fixed_width_column_wrapper<size_type>({0});

  EXPECT_THROW(cuspatial::polygon_intersect(x, y, space_offsets), cuspatial::logic_error);
}
