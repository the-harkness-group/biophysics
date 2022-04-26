#!/usr/bin/env python3

import os 
import sys
import yaml

# Get list of experiments to process
with open(sys.argv[1],'r') as stream:
    params = yaml.safe_load(stream)

folders = params['folders']
fidproc = params['fid.com']
nmrproc = params['nmrproc.com']
sparky_folder = params['sparky_folder']
spectrometer = params['spectrometer'] # Bruker or Varian
#if params['sparky_convert'] == 'n':
#    if os.path.isdir(f"./Analysis") == True:
#        pass
#    else:
#        os.makedirs('./Analysis')

if params['sparky_convert'] == 'y':
    if os.path.isdir(f"./{sparky_folder}") == False:
        os.makedirs(f"./{sparky_folder}")
    else:
        pass

sep = '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
# Process each experiment, need to make sure that fid.com and proc.com are in the current directory
for folder in folders:
        
        print(f"\n{sep}\nProcessing data in: {folder}\n{sep}")

        #folder_name = folder.split('/')[0] # Get experiment folder name
        folder_name = folder
        #folder_num = folder.split('/')[1] # Get spectrometer folder number, for Bruker!
        if spectrometer == 'Bruker':
            os.system(f"{fidproc} {folder}") # Run data conversion based on spectrometer format
        elif spectrometer == 'Varian':
            os.system(f"{fidproc} {folder}")
        os.system(f"{nmrproc} {folder}") # Process time-domain data and convert to frequency-domain
        
        if params['sparky_convert'] == 'y':
            os.system(f"/Applications/nmrfam-sparky-mac/NMRFAM-SPARKY.app/Contents/Resources/bin/pipe2ucsf ./{folder}/test.ft2 ./{folder}/{folder_name}.ucsf") # Convert processed data to sparky format
            os.system(f"cp ./{folder}/{folder_name}.ucsf ./{sparky_folder}") # Copy data to sparky folder for analysis
        
        #os.system(f"cp ./{folder}/test.ft2 ./Analysis/{folder_name}.ft2")
