#!/usr/bin/env python3

import os
import csv

basename = "inclusive_scan"
#variants = ["agency_acc_multicore", "agency_acc_tesla", "agency_cuda", "agency_omp", "agency_tbb", "thrust_cuda", "thrust_omp", "thrust_tbb", "intel_pstl"]
variants = ["agency_acc_tesla", "agency_cuda", "agency_omp", "agency_tbb", "thrust_cuda", "thrust_omp", "thrust_tbb", "intel_pstl"]
extension = ".csv"

spreadsheet = {}

for variant in variants:
    filename = os.path.abspath(basename + "_" + variant + extension)

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            size = int(row['size'])
            try:
                spreadsheet[size][variant] = float(row['bandwidth'])
            except KeyError:
                spreadsheet[size] = {}
                spreadsheet[size][variant] = float(row['bandwidth'])

print("size", *variants, sep=', ')

for size in sorted(spreadsheet.keys()):
    row = [size]
    for variant in variants:
        row.append(spreadsheet[size][variant])

    print(*row, sep=', ')

