AGENCY_INC_PATH := "/home/jhoberock/dev/git/agency-tot"

TBB2018_INC_PATH := "/opt/intel/tbb/include"
TBB2018_LIB_PATH := "/opt/intel/tbb/lib/intel64/gcc4.7"

INTELPSTL_INC_PATH := "/home/jhoberock/dev/git/parallelstl-intel/include"

INCLUDES := -I../.. -I$(AGENCY_INC_PATH) -I/usr/local/cuda/include

all: reduce_agency_cuda reduce_agency_acc_tesla reduce_agency_acc_multicore reduce_agency_omp reduce_agency_tbb reduce_thrust_cuda reduce_thrust_omp reduce_thrust_tbb reduce_intel_pstl

reduce_agency_cuda: reduce.cpp
	nvcc -DUSE_AGENCY_CUDA -O3 -std=c++11 --expt-extended-lambda $(INCLUDES) -x cu $^ -o $@.out

reduce_agency_acc_tesla: reduce.cpp
	pgc++ -DUSE_AGENCY_OPENACC -O3 -std=c++11 -ta:tesla,cc70,managed $(INCLUDES) $^ -o $@.out

reduce_agency_acc_multicore: reduce.cpp
	pgc++ -DUSE_AGENCY_OPENACC -O3 -std=c++11 -ta:multicore $(INCLUDES) $^ -o $@.out

reduce_agency_omp: reduce.cpp
	g++ -DUSE_AGENCY_OPENMP -fopenmp -O3 -std=c++11 $(INCLUDES) $^ -o $@.out

reduce_agency_tbb: reduce.cpp
	g++ -DUSE_AGENCY_TBB -fopenmp -O3 -std=c++11 $(INCLUDES) $^ -o $@.out -ltbb

reduce_thrust_cuda: reduce.cpp
	nvcc -DUSE_THRUST_CUDA -O3 -std=c++11 --expt-extended-lambda $(INCLUDES) -x cu $^ -o $@.out

reduce_thrust_omp: reduce.cpp
	g++ -DUSE_THRUST_OPENMP -fopenmp -O3 -std=c++11 $(INCLUDES) $^ -o $@.out

reduce_thrust_tbb: reduce.cpp
	g++ -DUSE_THRUST_TBB -fopenmp -O3 -std=c++11 $(INCLUDES) $^ -o $@.out -ltbb

reduce_intel_pstl: reduce.cpp
	icc -DUSE_INTEL_PSTL -O3 -std=c++11 -I$(TBB2018_INC_PATH) -I$(INTELPSTL_INC_PATH) $(INCLUDES) $^ -o $@.out -L$(TBB2018_LIB_PATH) -ltbb_preview

clean:
	rm -f *.o *.out

