#!/usr/bin/env python3

import sys
import os
import subprocess

sizes = [100, 128, 512, 1024, 10000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 8000000, 9000000, 10000000]

program_name = os.path.abspath(sys.argv[1])

print("size,bandwidth")

for size in sizes:
    output = subprocess.run([program_name, str(size)], stdout = subprocess.PIPE, stderr = subprocess.PIPE).stderr
    
    print(output.decode(), end = '')

