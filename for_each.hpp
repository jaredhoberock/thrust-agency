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


namespace experimental
{


template<class ExecutionPolicy, class Iterator, class Function>
Iterator for_each(ExecutionPolicy&& policy, Iterator first, Iterator last, Function f)
{
  using namespace agency;

  // XXX TODO: bake these requirements into .on() or agency::bulk_invoke()?
  auto ex = agency::require(policy.executor(), bulk, twoway);

  // XXX TODO: implement agency::prefer
  //auto ex = prefer(require(policy.executor(), bulk, twoway), always_blocking);

  auto n = std::distance(first, last);

  using index_type = executor_index_t<decltype(ex)>;

  using ignore_t = decltype(std::ignore);

  ex.bulk_twoway_execute(
    [=] __AGENCY_ANNOTATION (index_type idx, ignore_t&, ignore_t&) mutable
    {
      // XXX TODO: cast idx to iterator_difference

      f(first[idx]);
    },
    std::distance(first, last),
    []{ return std::ignore; },
    []{ return std::ignore; }
  ).wait();

  //using agent_type = typename std::decay<ExecutionPolicy>::type::execution_agent_type;

  //agency::bulk_invoke(policy(n), [=](agent_type& self) mutable
  //{
  //  f(first[self.rank()]);
  //});

  return first + n;
}


} // end test

