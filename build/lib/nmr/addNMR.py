#!/usr/bin/env python3
import os 
import sys
import yaml
import shutil

# Get experiment numbers or names
params = yaml.safe_load(open(sys.argv[1],'r'))
file_names = params['add_folders']
fidproc = params['fid.com']
nmrproc = params['nmrproc.com']
file_type = params['file_type']

print(f'\nThe NMR files that will be added together are: \n')
for fname in file_names:
       print(f"{fname}\n")

# Convert each experiments data
parent_dir = os.getcwd()
for fname in file_names:
    print(f"$$$$$$$$$$$$$$$$$ RUNNING BRUKER FID.COM {fname} $$$$$$$$$$$$$$$$$$$$$")
    os.system(f"{fidproc} {fname}")
    #os.system(f"{nmrproc} {fname}")
    print("\n")

# Add the fid data for the first and second file, then loop through and add the rest to the result
print(f"$$$$$$$$$$$$$$$$$ ADDING {file_names[0]}/test{file_type} AND {file_names[1]}/test{file_type}  >>> added.fid $$$$$$$$$$$$$$$$$$$$$")
os.system(f"addNMR -in1 {file_names[0]}/test{file_type} -in2 {file_names[1]}/test{file_type} -out added{file_type}")

for fname in file_names[2:]:
    print(f"$$$$$$$$$$$$$$$$$ ADDING added{file_type} AND {fname}/test{file_type} >>> added.fid $$$$$$$$$$$$$$$$$$$$$")
    os.system(f"addNMR -in1 added{file_type} -in2 {fname}/test{file_type} -out added{file_type}")
    
## Process each experiment's fid to frequency domain
#parent_dir = os.getcwd()
#for fname in file_names:
#    print(f"$$$$$$$$$$$$$$$$$ PROCESSING FIDS TO FREQUENCY DOMAIN {fname} $$$$$$$$$$$$$$$$$$$$$")
#    os.system(f"{nmrproc} {fname}")
#    print("\n")
#    
## Process each experiment's fid to frequency domain
#parent_dir = os.getcwd()
#for fname in file_names:
#    print(f"$$$$$$$$$$$$$$$$$ PROCESSING ADDED FIDS TO FREQUENCY DOMAIN {} $$$$$$$$$$$$$$$$$$$$$")
#    os.system(f"{nmrproc} {fname}")
#    print("\n")

# Put final added fid into the last file directory
print(f"$$$$$$$$$$$$$$$$$ MOVING added{file_type} FROM CURRENT DIRECTORY TO FINAL ADDING DIRECTORY ./{file_names[-1]}/{file_names[-1]}_added{file_type} $$$$$$$$$$$$$$$$$$$$$")
shutil.move(f"added{file_type}", f"./{file_names[-1]}/{file_names[-1]}_added{file_type}")