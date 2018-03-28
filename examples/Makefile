all: binary_transform_agency_acc binary_transform_agency_omp binary_transform_agency_tbb binary_transform_thrust_omp binary_transform_thrust_tbb

binary_transform_agency_acc: binary_transform.cpp
	pgc++ -DUSE_AGENCY_OPENACC -O3 -std=c++11 -ta:tesla,cc70,managed -I.. -I/usr/local/cuda/include $^ -o $@.out

binary_transform_agency_omp: binary_transform.cpp
	g++ -DUSE_AGENCY_OPENMP -fopenmp -O3 -std=c++11 -I.. -I/usr/local/cuda/include $^ -o $@.out

binary_transform_agency_tbb: binary_transform.cpp
	g++ -DUSE_AGENCY_TBB -fopenmp -O3 -std=c++11 -I.. -I/usr/local/cuda/include $^ -o $@.out -ltbb

binary_transform_thrust_omp: binary_transform.cpp
	g++ -DUSE_THRUST_OPENMP -fopenmp -O3 -std=c++11 -I.. -I/usr/local/cuda/include $^ -o $@.out

binary_transform_thrust_tbb: binary_transform.cpp
	g++ -DUSE_THRUST_TBB -fopenmp -O3 -std=c++11 -I.. -I/usr/local/cuda/include $^ -o $@.out -ltbb

clean:
	rm -f *.o *.out

