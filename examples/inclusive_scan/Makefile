AGENCY_INC_PATH := "/home/jhoberock/dev/git/agency-tot"

TBB2018_INC_PATH := "/opt/intel/tbb/include"
TBB2018_LIB_PATH := "/opt/intel/tbb/lib/intel64/gcc4.7"

INTELPSTL_INC_PATH := "/home/jhoberock/dev/git/parallelstl-intel/include"

INCLUDES := -I../.. -I$(AGENCY_INC_PATH) -I/usr/local/cuda/include

all: inclusive_scan_agency_cuda inclusive_scan_agency_acc_tesla inclusive_scan_agency_acc_multicore inclusive_scan_agency_omp inclusive_scan_agency_tbb inclusive_scan_thrust_cuda inclusive_scan_thrust_omp inclusive_scan_thrust_tbb inclusive_scan_intel_pstl

inclusive_scan_agency_cuda: inclusive_scan.cpp
	nvcc -DUSE_AGENCY_CUDA -O3 -std=c++11 --expt-extended-lambda $(INCLUDES) -x cu $^ -o $@.out

inclusive_scan_agency_acc_tesla: inclusive_scan.cpp
	pgc++ -DUSE_AGENCY_OPENACC -O3 -std=c++11 -ta:tesla,cc70,managed $(INCLUDES) $^ -o $@.out

inclusive_scan_agency_acc_multicore: inclusive_scan.cpp
	pgc++ -DUSE_AGENCY_OPENACC -O3 -std=c++11 -ta:multicore $(INCLUDES) $^ -o $@.out

inclusive_scan_agency_omp: inclusive_scan.cpp
	g++ -DUSE_AGENCY_OPENMP -fopenmp -O3 -std=c++11 $(INCLUDES) $^ -o $@.out

inclusive_scan_agency_tbb: inclusive_scan.cpp
	g++ -DUSE_AGENCY_TBB -fopenmp -O3 -std=c++11 $(INCLUDES) $^ -o $@.out -ltbb

inclusive_scan_thrust_cuda: inclusive_scan.cpp
	nvcc -DUSE_THRUST_CUDA -O3 -std=c++11 --expt-extended-lambda $(INCLUDES) -x cu $^ -o $@.out

inclusive_scan_thrust_omp: inclusive_scan.cpp
	g++ -DUSE_THRUST_OPENMP -fopenmp -O3 -std=c++11 $(INCLUDES) $^ -o $@.out

inclusive_scan_thrust_tbb: inclusive_scan.cpp
	g++ -DUSE_THRUST_TBB -fopenmp -O3 -std=c++11 $(INCLUDES) $^ -o $@.out -ltbb

inclusive_scan_intel_pstl: inclusive_scan.cpp
	icc -DUSE_INTEL_PSTL -O3 -std=c++11 -I$(TBB2018_INC_PATH) -I$(INTELPSTL_INC_PATH) $(INCLUDES) $^ -o $@.out -L$(TBB2018_LIB_PATH) -ltbb_preview

clean:
	rm -f *.o *.out

