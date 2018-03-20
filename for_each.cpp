#include <thrust/execution_policy.h>
#include <agency/agency.hpp>
#include <tuple>
#include <vector>

template<class ExecutionAgent, class Executor>
struct basic_thrust_policy : agency::basic_execution_policy<ExecutionAgent, Executor>,
                             thrust::execution_policy<basic_thrust_policy<ExecutionAgent, Executor>>
{
};


using ignore_t = decltype(std::ignore);


template<class ExecutionPolicy, class Iterator, class Function>
Iterator for_each(ExecutionPolicy policy, Iterator first, Iterator last, Function f)
{
  std::cout << "Hello, world from for_each(my_policy)!" << std::endl;

  using namespace agency;
  auto ex = require(policy.executor(), bulk, twoway);

  // XXX TODO: implement agency::prefer
  //auto ex = prefer(require(policy.executor(), bulk, twoway), always_blocking);

  auto n = std::distance(first, last);

  using index_type = executor_index_t<decltype(ex)>;

  ex.bulk_twoway_execute(
    [=](index_type idx, ignore_t&, ignore_t&)
    {
      // XXX TODO: cast idx to iterator_difference

      f(first[idx]);
    },
    std::distance(first, last),
    []{ return std::ignore; },
    []{ return std::ignore; }
  ).wait();

  return first + n;
}

int main()
{
  std::vector<int> vec(1);

  using my_policy = basic_thrust_policy<agency::parallel_agent, agency::parallel_executor>;
  my_policy policy;

  thrust::for_each(policy, vec.begin(), vec.end(), thrust::identity<int>());
}

