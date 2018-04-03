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

#include <thrust/for_each.h>
#include <thrust/tuple.h>

#include "basic_parallel_policy.hpp"
#include "acc_executor.hpp"


namespace experimental
{


template<class ForwardIterator1, class ForwardIterator2, class UnaryOperation>
ForwardIterator2 transform(const basic_parallel_policy<acc_executor>& policy,
                           ForwardIterator1 first, ForwardIterator1 last,
                           ForwardIterator2 result,
                           UnaryOperation unary_op)
{
  std::cout << "experimental::transform()" << std::endl;

  // zip up the ranges
  using iterator_tuple_type = thrust::tuple<ForwardIterator1, ForwardIterator2>;
  using zip_iterator_type = thrust::zip_iterator<iterator_tuple_type>;

  using reference_type = typename zip_iterator_type::reference;

  zip_iterator_type zipped_result =
    thrust::for_each(policy,
                     thrust::make_zip_iterator(thrust::make_tuple(first,result)),
                     thrust::make_zip_iterator(thrust::make_tuple(last,result)),
                     [=] __AGENCY_ANNOTATION (reference_type ref)
  {
    thrust::get<1>(ref) = unary_op(thrust::get<0>(ref));
  });

  return thrust::get<1>(zipped_result.get_iterator_tuple());
}


template<class ForwardIterator1, class ForwardIterator2, class ForwardIterator3, class BinaryOperation>
ForwardIterator3 transform(const basic_parallel_policy<acc_executor>& policy,
                           ForwardIterator1 first1, ForwardIterator1 last1,
                           ForwardIterator2 first2,
                           ForwardIterator3 result,
                           BinaryOperation binary_op)
{
  auto n = last1 - first1;
  auto ex = policy.executor();

  using ignored_t = decltype(std::ignore);
  ex.bulk_twoway_execute(
    [=] __AGENCY_ANNOTATION (size_t i, ignored_t&, ignored_t&)
    {
      result[i] = binary_op(first1[i], first2[i]);
    },
    n,
    []{ return std::ignore; },
    []{ return std::ignore; }
  ).wait();

  return result + n;
}


} // end test

