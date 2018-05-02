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

#include <agency/experimental/ranges/iterator_range.hpp>
#include <agency/experimental/ranges/tile.hpp>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>
#include <thread>


template<class Iterator1, class Iterator2, class T>
Iterator2 sequential_inclusive_scan(Iterator1 first, Iterator1 last, Iterator2 result, T init)
{
  if(first != last)
  {
    auto sum = init + *first;

    *result = sum;

    // XXX this loop yields nvbug 2102509
    for(++first, ++result; first != last; ++first, ++result)
      *result = sum = (sum + *first);
  }

  return result;
}


template<class Iterator1, class Iterator2, class T>
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


void inclusive_scan(const int* first, const int* last, int* result, int init)
{
  using namespace agency::experimental;

  size_t num_tiles = std::thread::hardware_concurrency();

  // create a view of the input
  auto input = make_iterator_range(first, last);

  // divide the input into a number of tiles approximately equal to the executor's unit_shape
  auto input_tiles = tile_evenly(input, num_tiles);

  // phase 1: for each tile, compute its sum
  std::vector<int> carries(input_tiles.size());
  #pragma acc parallel loop
  for(size_t i = 0; i < input_tiles.size(); ++i)
  {
    carries[i] = std::accumulate(input_tiles[i].begin(), input_tiles[i].end(), 0, std::plus<int>());
  }

  // phase 2: exclusive_scan the sums to turn them into carry-ins for phase 3
  sequential_exclusive_scan(carries.begin(), carries.end(), carries.begin(), init);

  // phase 3: inclusive_scan to the result, using the carries as initializers
  auto output = make_iterator_range(result, result + (last - first));
  auto output_tiles = tile_evenly(output, num_tiles);
  assert(output_tiles.size() == input_tiles.size());
  #pragma acc parallel loop
  for(size_t i = 0; i < input_tiles.size(); ++i)
  {
    auto input_tile = input_tiles[i];
    auto output_tile = output_tiles[i];

    sequential_inclusive_scan(input_tile.begin(), input_tile.end(), output_tile.begin(), carries[i]);
  }
}


double test(size_t n)
{
  // set up some random inputs
  std::vector<int> x(n, 1);
  std::generate(x.begin(), x.end(), std::default_random_engine());

  // storage for the result
  std::vector<int> result(n);

  // run once
  inclusive_scan(x.data(), x.data() + x.size(), result.data(), 0);

  // check the result
  std::vector<int> expected_result(n);
  std::partial_sum(x.begin(), x.end(), expected_result.begin());
  assert(expected_result == result);

  // time a number of trials
  size_t num_trials = 100;

  auto start = std::chrono::high_resolution_clock::now();
  for(size_t i = 0; i < num_trials; ++i)
  {
    inclusive_scan(x.data(), x.data() + x.size(), result.data(), 0);
  }
  std::chrono::duration<double> elapsed_seconds = std::chrono::high_resolution_clock::now() - start;

  auto mean_seconds = elapsed_seconds.count() / num_trials;
  double gigabytes = double(2 * n * sizeof(int)) / (1 << 30);
  return gigabytes / mean_seconds;
}


int main(int argc, char** argv)
{
  size_t n = 8 << 20;
  if(argc > 1)
  {
    n = std::atoi(argv[1]);
  }

  double bandwidth = test(n);

  std::clog << n << ", " << bandwidth << std::endl;

  std::cout << "Binary transform bandwidth: " << bandwidth << " GB/s" << std::endl;

  std::cout << "OK" << std::endl;
}

