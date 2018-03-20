#include <thrust/execution_policy.h>
#include <agency/agency.hpp>
#include <tuple>
#include <vector>
#include <cassert>

#include "acc.hpp"


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


using ignore_t = decltype(std::ignore);


template<class ExecutionPolicy, class Iterator, class Function>
Iterator for_each(ExecutionPolicy&& policy, Iterator first, Iterator last, Function f)
{
  std::cout << "Hello, world from for_each(my_policy)!" << std::endl;

  using namespace agency;

  // XXX TODO: bake these requirements into .on() or agency::bulk_invoke()?
  auto ex = agency::require(policy.executor(), bulk, twoway);

  // XXX TODO: implement agency::prefer
  //auto ex = prefer(require(policy.executor(), bulk, twoway), always_blocking);

  auto n = std::distance(first, last);

  using index_type = executor_index_t<decltype(ex)>;

  ex.bulk_twoway_execute(
    [=](index_type idx, ignore_t&, ignore_t&) mutable
    {
      // XXX TODO: cast idx to iterator_difference

      f(first[idx]);
    },
    std::distance(first, last),
    []{ return std::ignore; },
    []{ return std::ignore; }
  ).wait();

  //using agent_type = typename ExecutionPolicy::execution_agent_type;

  //agency::bulk_invoke(policy(n), [=](agent_type& self)
  //{
  //  f(first[self.rank()]);
  //});

  return first + n;
}

int main()
{
  std::vector<int> vec(1, 13);

  basic_parallel_policy<agency::parallel_executor> par;
  
  // when we call thrust::for_each with our custom execution policy, it will find the overload of for_each above
  thrust::for_each(par.on(acc_executor()), vec.begin(), vec.end(), thrust::identity<int>());

  // when we call thrust::transform with our custom execution policy, it will implement the transform via our overload of for_each above
  thrust::transform(par.on(acc_executor()), vec.begin(), vec.end(), vec.begin(), thrust::negate<int>());

  assert(vec[0] == -13);
}

