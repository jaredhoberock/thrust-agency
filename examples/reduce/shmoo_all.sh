#!/bin/bash -v

./shmoo_reduce.py reduce_agency_acc_multicore.out > reduce_agency_acc_multicore.csv
./shmoo_reduce.py reduce_agency_acc_tesla.out > reduce_agency_acc_tesla.csv
./shmoo_reduce.py reduce_agency_cuda.out > reduce_agency_cuda.csv
./shmoo_reduce.py reduce_agency_omp.out > reduce_agency_omp.csv
./shmoo_reduce.py reduce_agency_tbb.out > reduce_agency_tbb.csv
./shmoo_reduce.py reduce_thrust_cuda.out > reduce_thrust_cuda.csv
./shmoo_reduce.py reduce_thrust_omp.out > reduce_thrust_omp.csv
./shmoo_reduce.py reduce_thrust_tbb.out > reduce_thrust_tbb.csv
