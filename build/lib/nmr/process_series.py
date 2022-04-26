#!/usr/bin/env python3
import os 
import sys
import yaml

def main():
    # Get list of experiments to process
    params = yaml.safe_load(open(sys.argv[1],'r'))
    # Process experiment series using NMRPipe
    process_series(params)

def process_series(params):

    # Get processing parameters from input
    folders = params['Experiments']
    nmrproc = params['Processing script']
    
    if params['Sparky convert'] == 'y':
        sparky_folder = params['Sparky folder']
        if os.path.isdir(f"./{sparky_folder}") == False:
            os.makedirs(f"./{sparky_folder}")
        else:
            pass

    sep = '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
    # Process each experiment, need to make sure that fid.com and proc.com are in the current directory
    for folder in folders:
        print(f"\n{sep}\nProcessing data in: {folder}\n{sep}")
        folder_name = folder
        os.system(f"{nmrproc} {folder}") # Process time-domain data and convert to frequency-domain
        
        if params['Sparky convert'] == 'y':
            os.system(f"/Applications/nmrfam-sparky-mac/NMRFAM-SPARKY.app/Contents/Resources/bin/pipe2ucsf ./{folder}/test.ft2 ./{folder}/{folder_name}.ucsf") # Convert processed data to sparky format
            os.system(f"cp ./{folder}/{folder_name}.ucsf ./{sparky_folder}") # Copy data to sparky folder for analysis

main()
