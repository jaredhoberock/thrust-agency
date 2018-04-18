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
#include <agency/bulk_invoke.hpp>
#include <agency/experimental/ranges/iterator_range.hpp>
#include <agency/experimental/ranges/tile.hpp>
#include <agency/execution/executor/customization_points/unit_shape.hpp>
#include <agency/container/vector.hpp>
#include <thrust/reduce.h>

#include "execution_policy_allocator.hpp"
#include "acc_executor.hpp"
#include <agency/execution/executor/require.hpp>
#include <agency/execution/executor/properties.hpp>

#include <atomic>


namespace experimental
{


template<class Iterator, class T, class BinaryOperation>
T reduce(basic_parallel_policy<acc_executor> policy, Iterator first, Iterator last, T init, BinaryOperation binary_op)
{
  // executor-based implementation
  // XXX 247 GB/s

  using namespace agency;
  using namespace agency::experimental;

  // create a view of the input
  auto input = make_iterator_range(first, last);

  // XXX do a cyclic view before tiling?

  // divide the input into a number of tiles approximately equal to the executor's unit_shape
  auto tiles = tile_evenly(input, unit_shape(policy.executor()));

  using partial_sums_type = std::vector<T>;

  auto ex = agency::require(policy.executor(), bulk, twoway);

  int num_partial_sums = tiles.size();

  partial_sums_type partial_sums = ex.bulk_twoway_execute(
    [=](size_t i, partial_sums_type& result, int) {
      // get this agent's tile
      auto this_tile = tiles[i];

      // return the sum of this tile
      result[i] = thrust::reduce(thrust::seq, this_tile.begin() + 1, this_tile.end(), this_tile[0], binary_op);
    },
    num_partial_sums,
    [=]{ return partial_sums_type(num_partial_sums); },
    []{ return 0; }
  ).get();

  // return the sum of partial sums
  return thrust::reduce(thrust::seq, partial_sums.begin(), partial_sums.end(), init, binary_op);

//  // raw OpenACC implementation
//  // XXX 595 GB/s
//  size_t n = last - first;
//
//  #pragma acc parallel loop reduction(+:init)
//  for(size_t i = 0; i < n; ++i)
//  {
//    init += first[i];
//  }
//
//  return init;
}


template<class ExecutionPolicy, class Iterator, class T, class BinaryOperation>
T reduce(ExecutionPolicy&& policy, Iterator first, Iterator last, T init, BinaryOperation binary_op)
{
  // bulk_invoke implementation
  using namespace agency;
  using namespace agency::experimental;

  // XXX require/prefer properties here

  // create a view of the input
  auto input = make_iterator_range(first, last);

  // divide the input into a number of tiles equal to the executor's unit_shape
  auto tiles = tile_evenly(input, unit_shape(policy.executor()));

  // compute the sum of each tile separately, in parallel
  using agent_type = typename std::decay<ExecutionPolicy>::type::execution_agent_type;
  auto partial_sums = bulk_invoke(policy(tiles.size()), [=] __AGENCY_ANNOTATION (agent_type& self)
  {
    // get this agent's tile
    auto this_tile = tiles[self.index()];

    // return the sum of this tile
    return thrust::reduce(thrust::seq, this_tile.begin() + 1, this_tile.end(), this_tile[0], binary_op);
  });

  // return the sum of partial sums
  return thrust::reduce(thrust::seq, partial_sums.begin(), partial_sums.end(), init, binary_op);
}


} // end test

