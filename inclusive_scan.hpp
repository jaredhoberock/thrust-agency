// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "basic_parallel_policy.hpp"
#include <agency/execution/executor/require.hpp>
#include <agency/execution/executor/properties.hpp>
#include <agency/execution/executor/customization_points/unit_shape.hpp>
#include <agency/experimental/ranges/iterator_range.hpp>
#include <agency/experimental/ranges/tile.hpp>
#include <agency/container/vector.hpp>
#include <type_traits>
#include <iterator>

#include "execution_policy_allocator.hpp"
#include "basic_parallel_policy.hpp"
#include "acc_executor.hpp"


namespace experimental
{
namespace detail
{


template<class Iterator, class T, class BinaryOperation>
__AGENCY_ANNOTATION
T sequential_reduce(Iterator first, Iterator last, T init, BinaryOperation binary_op)
{
  for(; first != last; ++first)
  {
    init = binary_op(init, *first);
  }

  return init;
}


template<class Iterator1, class Iterator2, class T>
__AGENCY_ANNOTATION
Iterator2 sequential_inclusive_scan(Iterator1 first, Iterator1 last, Iterator2 result, T init)
{
  if(first != last)
  {
    auto sum = init + *first;

    *result = sum;

    for(++first, ++result; first != last; ++first, ++result)
      *result = sum = (sum + *first);
  }

  return result;
}


template<class Iterator1, class Iterator2, class T>
__AGENCY_ANNOTATION
Iterator2 sequential_exclusive_scan(Iterator1 first, Iterator1 last, Iterator2 result, T init)
{
  if(first != last)
  {
    auto tmp = *first;  // temporary value allows in-situ scan
    auto sum = init;

    *result = sum;
    sum = sum + tmp;

    for(++first, ++result; first != last; ++first, ++result)
    {
      tmp = *first;
      *result = sum;
      sum = sum + tmp;
    }
  }

  return result;
}


} // end detail


// XXX WAR nvbug 2159271
template<class Iterator1, class Iterator2, class BinaryOperation, class T>
Iterator2 inclusive_scan(basic_parallel_policy<acc_executor> policy, Iterator1 first, Iterator1 last, Iterator2 result, BinaryOperation binary_op, T init)
{
  // create a view of the input
  auto input = agency::experimental::make_iterator_range(first, last);

  // divide the input into a number of tiles approximately equal to the executor's unit_shape
  auto input_tiles = tile_evenly(input, agency::unit_shape(policy.executor()));
  size_t num_tiles = input_tiles.size();

  // phase 1: for each tile, compute its sum
  using carry_type = typename std::result_of<
    BinaryOperation(typename std::iterator_traits<Iterator1>::value_type, typename std::iterator_traits<Iterator1>::value_type)
  >::type;

  using carries_type = agency::vector<carry_type>;

  auto ex = agency::require(policy.executor(), agency::bulk, agency::twoway);

  // phase 1: for each tile, compute its sum
  carries_type carries(num_tiles);
  #pragma acc parallel loop
  for(size_t i = 0; i < num_tiles; ++i)
  {
    // get this agent's tile
    auto this_tile = input_tiles[i];

    // return the sum of this tile
    carries[i] = detail::sequential_reduce(this_tile.begin() + 1, this_tile.end(), this_tile[0], binary_op);
  }

  // phase 2: exclusive_scan the sums to turn them into carry-ins for phase 3
  auto carries_begin = carries.begin();
  auto carries_end = carries.end();
  detail::sequential_exclusive_scan(carries_begin, carries_end, carries_begin, init);

  // phase 3: inclusive_scan to the result, using the carries as initializers
  auto output = agency::experimental::make_iterator_range(result, result + (last - first));
  auto output_tiles = agency::experimental::tile_evenly(output, num_tiles);
  assert(output_tiles.size() == input_tiles.size());

  #pragma acc parallel loop
  for(size_t i = 0; i < num_tiles; ++i)
  {
    auto input_tile = input_tiles[i];
    auto output_tile = output_tiles[i];

    detail::sequential_inclusive_scan(input_tile.begin(), input_tile.end(), output_tile.begin(), carries_begin[i]);
  }

  return output.end();
}


template<class ExecutionPolicy, class Iterator1, class Iterator2, class BinaryOperation, class T>
Iterator2 inclusive_scan(ExecutionPolicy&& policy, Iterator1 first, Iterator1 last, Iterator2 result, BinaryOperation binary_op, T init)
{
  // create a view of the input
  auto input = agency::experimental::make_iterator_range(first, last);

  // divide the input into a number of tiles approximately equal to the executor's unit_shape
  auto input_tiles = tile_evenly(input, agency::unit_shape(policy.executor()));
  size_t num_tiles = input_tiles.size();

  // phase 1: for each tile, compute its sum
  using carry_type = typename std::result_of<
    BinaryOperation(typename std::iterator_traits<Iterator1>::value_type, typename std::iterator_traits<Iterator1>::value_type)
  >::type;

  using allocator_type = execution_policy_allocator_t<typename std::decay<ExecutionPolicy>::type, carry_type>;
  using carries_type = agency::vector<carry_type, allocator_type>;

  auto ex = agency::require(policy.executor(), agency::bulk, agency::twoway);

  // phase 1: for each tile, compute its sum
  carries_type carries = ex.bulk_twoway_execute(
    [=] __AGENCY_ANNOTATION (size_t i, carries_type& carries, int)
    {
      // get this agent's tile
      auto this_tile = input_tiles[i];

      // return the sum of this tile
      carries[i] = detail::sequential_reduce(this_tile.begin() + 1, this_tile.end(), this_tile[0], binary_op);
    },
    num_tiles,
    [=]{ return carries_type(num_tiles); },
    []{ return 0; }
  ).get();

  // phase 2: exclusive_scan the sums to turn them into carry-ins for phase 3
  auto carries_begin = carries.begin();
  auto carries_end = carries.end();
  ex.bulk_twoway_execute(
    [=] __AGENCY_ANNOTATION (size_t, int, int)
    {
      detail::sequential_exclusive_scan(carries_begin, carries_end, carries_begin, init);
    },
    1,
    []{ return 0; },
    []{ return 0; }
  ).wait();

  // phase 3: inclusive_scan to the result, using the carries as initializers
  auto output = agency::experimental::make_iterator_range(result, result + (last - first));
  auto output_tiles = agency::experimental::tile_evenly(output, num_tiles);
  assert(output_tiles.size() == input_tiles.size());
  ex.bulk_twoway_execute(
    [=] __AGENCY_ANNOTATION (size_t i, int, int)
    {
      auto input_tile = input_tiles[i];
      auto output_tile = output_tiles[i];

      detail::sequential_inclusive_scan(input_tile.begin(), input_tile.end(), output_tile.begin(), carries_begin[i]);
    },
    num_tiles,
    []{ return 0; },
    []{ return 0; }
  ).wait();

  return output.end();
}


template<class ExecutionPolicy, class Iterator1, class Iterator2, class BinaryOperation>
Iterator2 inclusive_scan(ExecutionPolicy&& policy, Iterator1 first, Iterator1 last, Iterator2 result, BinaryOperation binary_op)
{
  return experimental::inclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, result, binary_op, typename std::iterator_traits<Iterator1>::value_type{});
}


} // end experimental

