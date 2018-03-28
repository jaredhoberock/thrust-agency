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

#include <agency/future/always_ready_future.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/execution_categories.hpp>

// TBB doesn't support the PGI compiler
#if !defined(__PGI)
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

class tbb_executor
{
  public:
    using execution_category = agency::parallel_execution_tag;

    template<class T>
    using future = agency::always_ready_future<T>;

  private:
    template<class Function, class Result, class SharedParameter>
    struct body
    {
      mutable Function f;
      Result& result;
      SharedParameter& shared_parm;

#if !defined(__PGI)
      void operator()(const tbb::blocked_range<size_t>& r) const
      {
        for(auto i = r.begin(); i != r.end(); ++i)
        {
          f(i, result, shared_parm);
        }
      }
#endif
    };

  public:
    template<class Function, class ResultFactory, class SharedFactory>
    future<agency::detail::result_of_t<ResultFactory()>>
      bulk_twoway_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
    {
      auto result = result_factory();
      auto shared_parm = shared_factory();

#if !defined(__PGI)
      tbb::parallel_for(tbb::blocked_range<size_t>(0,n), body<Function,decltype(result),decltype(shared_parm)>{f, result, shared_parm});
#else
      static_assert(sizeof(Function) && false, "tbb_executor::bulk_twoway_execute(): TBB does not support the PGI compiler.");
#endif

      return agency::make_always_ready_future(std::move(result));
    }
};

