#!/bin/bash -v

./shmoo_binary_transform.py binary_transform_agency_acc.out > binary_transform_agency_acc.csv
./shmoo_binary_transform.py binary_transform_agency_cuda.out > binary_transform_agency_cuda.csv
./shmoo_binary_transform.py binary_transform_agency_omp.out > binary_transform_agency_omp.csv
./shmoo_binary_transform.py binary_transform_agency_tbb.out > binary_transform_agency_tbb.csv
./shmoo_binary_transform.py binary_transform_thrust_cuda.out > binary_transform_thrust_cuda.csv
./shmoo_binary_transform.py binary_transform_thrust_omp.out > binary_transform_thrust_omp.csv
./shmoo_binary_transform.py binary_transform_thrust_tbb.out > binary_transform_thrust_tbb.csv
