#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 09:52:18 2020

@author: robertharkness
"""

import os 
import sys
import yaml
import shutil

# Get experiment numbers or names
params = yaml.safe_load(open(sys.argv[1],'r'))
first_file = params['first file']
#nmrproc = params['nmrproc.com']
file_type = params['file type']
file_prefix = params['file prefix']
number_blocks = params['number blocks']
block_size = params['block size']

# Add FIDs and process each added block
file_index = first_file + 1
for x in range(1,number_blocks + 1): # Iterate through number of FID blocks to be generated
    
    for y in range(2,block_size + 1): # Iterate through FIDs in each block and add them together
        
        if y == 2: # Add the data for the first and second file in each block
            
            print(f"$$$$$$$$$$$$$$$$$ ADDING {file_prefix + str(file_index - 1) + '.fid'} AND {file_prefix + str(file_index) + '.fid'} $$$$$$$$$$$$$$$$$$$$$")
            
            os.system(f"addNMR -in1 {file_prefix + str(file_index - 1) + '.fid'} -in2 {file_prefix + str(file_index) + '.fid'} -out {'added' + str(x) + '.fid'}")
            
            file_index += 1
            
        if y > 2: # Loop through rest of FIDs in block and add these to the output generated from adding the first two FIDs in the block
            
            print(f"$$$$$$$$$$$$$$$$$ ADDING {'added' + str(x) + '.fid'} AND {file_prefix + str(file_index) + '.fid'} $$$$$$$$$$$$$$$$$$$$$")
            
            os.system(f"addNMR -in1 {'added' + str(x) + '.fid'} -in2 {file_prefix + str(file_index) + '.fid'} -out {'added' + str(x) + '.fid'}")
            
            file_index += 1
    
    file_index +=1
    
    #print(f"$$$$$$$$$$$$$$$$$ PROCESSING {added + str(x) + file_type} $$$$$$$$$$$$$$$$$$$$$") # Process added FID blocks
    #os.system(f"{nmrproc} {added + str(x) + file_type}")
    
    os.system(f"nmrproc.com {'added' + str(x) + '.fid'}") # Process added FID block
    
    #shutil.move(f"{'added' + str(x) + '.fid'}", f"../addedfids/") # Put added FID block into added FIDs directory
    shutil.move(f"{'added' + str(x) + '.fid.ft'}", f"../addedfids/") # Put processed added FID block into added FIDs directory
