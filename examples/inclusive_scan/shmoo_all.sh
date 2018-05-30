#!/bin/bash -v

./shmoo_inclusive_scan.py inclusive_scan_agency_acc_multicore.out > inclusive_scan_agency_acc_multicore.csv
./shmoo_inclusive_scan.py inclusive_scan_agency_acc_tesla.out > inclusive_scan_agency_acc_tesla.csv
./shmoo_inclusive_scan.py inclusive_scan_agency_cuda.out > inclusive_scan_agency_cuda.csv
./shmoo_inclusive_scan.py inclusive_scan_agency_omp.out > inclusive_scan_agency_omp.csv
./shmoo_inclusive_scan.py inclusive_scan_agency_tbb.out > inclusive_scan_agency_tbb.csv
./shmoo_inclusive_scan.py inclusive_scan_thrust_cuda.out > inclusive_scan_thrust_cuda.csv
./shmoo_inclusive_scan.py inclusive_scan_thrust_omp.out > inclusive_scan_thrust_omp.csv
./shmoo_inclusive_scan.py inclusive_scan_thrust_tbb.out > inclusive_scan_thrust_tbb.csv
./shmoo_inclusive_scan.py inclusive_scan_intel_pstl.out > inclusive_scan_intel_pstl.csv
