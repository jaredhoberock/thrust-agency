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

#include <agency/agency.hpp>
#include <agency/cuda/execution/executor/parallel_executor.hpp>
#include <agency/omp/execution.hpp>
#include <thrust/transform.h>
#include <thrust/system/omp/execution_policy.h>
#include <agency/container/vector.hpp>
#include <cassert>
#include <iostream>
#include <chrono>

#include "reduce.hpp"
#include "acc_executor.hpp"
#include "tbb_executor.hpp"
#include "execution_policy_allocator.hpp"

#ifdef USE_INTEL_PSTL
#include <pstl/execution>
#include <pstl/numeric>
#endif // USE_INTEL_PSTL


template<class ExecutionPolicy>
int sum(ExecutionPolicy&& policy, size_t n, const int* x)
{
  return thrust::reduce(policy, x, x + n);
}

#ifdef USE_INTEL_PSTL
int sum(pstl::execution::v1::parallel_policy policy, size_t n, const int* x)
{
  return std::reduce(policy, x, x + n);
}


template<class T>
struct execution_policy_allocator<pstl::execution::v1::parallel_policy, T>
{
  using type = std::allocator<T>;
};

#endif // USE_INTEL_PSTL


template<class ExecutionPolicy>
double test(ExecutionPolicy policy, size_t n)
{
  using allocator_type = execution_policy_allocator_t<ExecutionPolicy, int>;

  // set up some inputs
  agency::vector<int, allocator_type> x(n, 1);

  // run once
  int result = sum(policy, n, x.data());

  // check the result
  if(n != result)
  {
    std::cerr << "Unexpected result: " << result << ", expected " << n << std::endl;
  }
  assert(n == result);

  // time a number of trials
  size_t num_trials = 100;

  auto start = std::chrono::high_resolution_clock::now();
  for(size_t i = 0; i < num_trials; ++i)
  {
    sum(policy, n, x.data());
  }
  std::chrono::duration<double> elapsed_seconds = std::chrono::high_resolution_clock::now() - start;

  auto mean_seconds = elapsed_seconds.count() / num_trials;
  double gigabytes = double(n * sizeof(int)) / (1 << 30);
  return gigabytes / mean_seconds;
}


int main(int argc, char** argv)
{
  // select a policy based on compilation environment
  auto policy = 
#if defined(USE_AGENCY_CUDA)
    experimental::basic_parallel_policy<agency::parallel_executor>().on(agency::cuda::parallel_executor())
#elif defined(USE_AGENCY_OPENACC)
    experimental::basic_parallel_policy<agency::parallel_executor>().on(acc_executor())
#elif defined(USE_AGENCY_OPENMP)
    experimental::basic_parallel_policy<agency::parallel_executor>().on(agency::omp::parallel_executor())
#elif defined(USE_AGENCY_TBB)
    experimental::basic_parallel_policy<agency::parallel_executor>().on(tbb_executor())
#elif defined(USE_THRUST_CUDA)
    thrust::cuda::par
#elif defined(USE_THRUST_OPENMP)
    thrust::omp::par
#elif defined(USE_THRUST_TBB)
    thrust::tbb::par
#elif defined(USE_INTEL_PSTL)
    pstl::execution::v1::par
#else
    experimental::basic_parallel_policy<agency::parallel_executor>()
#endif
  ;

  size_t n = 8 << 20;
  if(argc > 1)
  {
    n = std::atoi(argv[1]);
  }

  double bandwidth = test(policy, n);

  std::clog << n << ", " << bandwidth << std::endl;

  std::cout << "Binary transform bandwidth: " << bandwidth << " GB/s" << std::endl;

  std::cout << "OK" << std::endl;
}

