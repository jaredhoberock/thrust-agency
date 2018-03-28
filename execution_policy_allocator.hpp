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

#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/execution_policy/execution_policy_traits.hpp>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <agency/cuda/memory/allocator/allocator.hpp>
#include <memory>

// TBB does not support the PGI compiler
#ifndef __PGI
#include <thrust/system/tbb/execution_policy.h>
#endif // __PGI


// a portable way to get a suitable allocator given an execution policy
template<class ExecutionPolicy, class T>
struct execution_policy_allocator;

template<class ExecutionPolicy, class T>
struct execution_policy_allocator
{
  // for Agency execution policies, we can just ask
  // the associated executor which allocator it prefers
  using type = agency::executor_allocator_t<
    typename ExecutionPolicy::executor_type,
    T
  >;
};

template<class ExecutionPolicy, class T>
using execution_policy_allocator_t = typename execution_policy_allocator<ExecutionPolicy,T>::type;


// introduce specializations for Thrust execution policies
template<class T>
struct execution_policy_allocator<typename std::decay<decltype(thrust::cuda::par)>::type, T>
{
  using type = agency::cuda::allocator<T>;
};

template<class T>
struct execution_policy_allocator<typename std::decay<decltype(thrust::omp::par)>::type, T>
{
  using type = std::allocator<T>;
};


// TBB does not support the PGI compiler
#ifndef __PGI
template<class T>
struct execution_policy_allocator<typename std::decay<decltype(thrust::tbb::par)>::type, T>
{
  using type = std::allocator<T>;
};
#endif // __PGI

