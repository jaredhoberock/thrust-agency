#include <thrust/execution_policy.h>
#include <agency/agency.hpp>
#include <tuple>
#include <vector>
#include <cassert>

#include "acc.hpp"

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
Iterator for_each(ExecutionPolicy policy, Iterator first, Iterator last, Function f)
{
  std::cout << "Hello, world from for_each(my_policy)!" << std::endl;

  using namespace agency;
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
  
  thrust::for_each(par.on(acc_executor()), vec.begin(), vec.end(), thrust::identity<int>());

  thrust::transform(par.on(acc_executor()), vec.begin(), vec.end(), vec.begin(), thrust::negate<int>());

  assert(vec[0] == -13);
}

