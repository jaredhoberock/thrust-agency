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

#include <agency/agency.hpp>
#include <thrust/execution_policy.h>


namespace experimental
{


// to compose with Thrust, we need to define an execution policy type:
//
// 1. derive from agency::basic_execution_policy to get conveniences like .on(), .executor(), etc. Thrust is oblivious to these conveniences.
// 2. derive from thrust::execution_policy to hook into Thrust's algorithm dispatch machinery. This is the part Thrust cares about.
//    XXX it would be nice not to have to derive from thrust::execution_policy and instead simply define overloads for the algorithms we're interested in customizing, as we do below
template<class Executor>
class basic_parallel_policy : public agency::basic_execution_policy<agency::parallel_agent, Executor, basic_parallel_policy<Executor>>,
                              public thrust::execution_policy<basic_parallel_policy<Executor>>
{
  private:
    using super_t = agency::basic_execution_policy<agency::parallel_agent, Executor, basic_parallel_policy<Executor>>;

  public:
    using super_t::super_t;

    template<class ReplacementExecutor>
    basic_parallel_policy<ReplacementExecutor> replace_executor(const ReplacementExecutor& ex) const
    {
      return basic_parallel_policy<ReplacementExecutor>(this->param(), ex);
    }
};


} // end experimental

