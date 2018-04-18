AGENCY_INC_PATH := "/home/jhoberock/dev/git/agency-tot"

INCLUDES := -I../.. -I$(AGENCY_INC_PATH)

all: reduce_agency_cuda reduce_agency_acc_tesla reduce_agency_acc_multicore reduce_agency_omp reduce_agency_tbb reduce_thrust_cuda reduce_thrust_omp reduce_thrust_tbb

reduce_agency_cuda: reduce.cpp
	nvcc -DUSE_AGENCY_CUDA -O3 -std=c++11 --expt-extended-lambda $(INCLUDES) -x cu $^ -o $@.out

reduce_agency_acc_tesla: reduce.cpp
	pgc++ -DUSE_AGENCY_OPENACC -O3 -std=c++11 -ta:tesla,cc70,managed $(INCLUDES) -I/usr/local/cuda/include $^ -o $@.out

reduce_agency_acc_multicore: reduce.cpp
	pgc++ -DUSE_AGENCY_OPENACC -O3 -std=c++11 -ta:multicore $(INCLUDES) -I/usr/local/cuda/include $^ -o $@.out

reduce_agency_omp: reduce.cpp
	g++ -DUSE_AGENCY_OPENMP -fopenmp -O3 -std=c++11 $(INCLUDES) -I/usr/local/cuda/include $^ -o $@.out

reduce_agency_tbb: reduce.cpp
	g++ -DUSE_AGENCY_TBB -fopenmp -O3 -std=c++11 $(INCLUDES) -I/usr/local/cuda/include $^ -o $@.out -ltbb

reduce_thrust_cuda: reduce.cpp
	nvcc -DUSE_THRUST_CUDA -O3 -std=c++11 --expt-extended-lambda $(INCLUDES) -x cu $^ -o $@.out

reduce_thrust_omp: reduce.cpp
	g++ -DUSE_THRUST_OPENMP -fopenmp -O3 -std=c++11 $(INCLUDES) -I/usr/local/cuda/include $^ -o $@.out

reduce_thrust_tbb: reduce.cpp
	g++ -DUSE_THRUST_TBB -fopenmp -O3 -std=c++11 $(INCLUDES) -I/usr/local/cuda/include $^ -o $@.out -ltbb

clean:
	rm -f *.o *.out

