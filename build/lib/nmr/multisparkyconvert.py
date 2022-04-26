#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 16:23:04 2020

@author: robertharkness
"""

import sys
import yaml
import os

# Convert multiple nmrpipe files to sparky .ucsf files
def main():
    
    params = yaml.safe_load(open(sys.argv[1],'r')) # Read parameters to get files for conversion
    
    params['Sparky Files'] = [] # Set up new sparky filename list
    for file in params['NMRPipe Files']:
        
        sparky_filename = file.split('.')[0] + '.ucsf' # Get new sparky filename by removing NMRPipe suffix and appending sparky suffix
        os.system(f"/Applications/nmrfam-sparky-mac/NMRFAM-SPARKY.app/Contents/Resources/bin/pipe2ucsf {file} {sparky_filename}")
        
        params['Sparky Files'].append(sparky_filename) # Write new sparky filenames to params
    
    with open('params.yaml','w') as newparamsfile: # Overwrite parameter yaml file to include new sparky filenames
        yaml.dump(params,newparamsfile)
        
main()