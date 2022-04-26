#!/usr/bin/env python3

import os 
import sys

# Get experiment numbers
files = [i for i in range(int(sys.argv[1]),int(sys.argv[2])+1)]

print(f'The NMR files that will be added together are: {files}')

# Process each experiment, need to make sure that fid.com and proc.com are in the current directory
for fname in files:
    os.system(f"./fid.com {fname}")
    os.system(f"./proc.com {fname}")

# Get second file number
second_file = str(int(sys.argv[1]) + 1)

# Add the data for the first and second file, then loop through and add the rest to the result
os.system("addNMR -in1 {}/test.ft2 -in2 {}/test.ft2 -out added.ft2".format(sys.argv[1], second_file))

for f in files[2:]:
    os.system("addNMR -in1 added.ft2 -in2 %d/test.ft2 -out added.ft2"%f)

