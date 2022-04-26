#!/usr/bin/env python3
import os 
import sys
import yaml
import shutil

# Read parameters for processing a titration
def main():
    # Get experiment numbers or names
    params = yaml.safe_load(open(sys.argv[1],'r'))
    #file_names = params['add_folders']
    fidproc = params['fid.com']
    #nmrproc = params['nmrproc.com']
    file_type = params['file_type']

    conversion(params, fidproc)
    addition(params, file_type)

# Convert NMR data to nmrpipe files
def conversion(titration_points, fidproc):
    # Convert each experiments data
    parent_dir = os.getcwd()

    for k in titration_points.keys():
        for fname in titration_points[k]:
            print(f"$$$$$$$$$$$$$$$$$ RUNNING BRUKER FID.COM {fname} $$$$$$$$$$$$$$$$$$$$$")
            os.system(f"{fidproc} {fname}")
            print("\n")

# Add nmrpipe files together for the titration points whose data has been collected in blocks
def addition(params, file_type):

    for k in params.keys():

        file_names = params[k]

        if len(file_names) == 1: # Don't add because only 1 file
            print(f"$$$$$$$$$$$$$$$$$ NO ADDING FOR {file_names[0]}/test{file_type} REQUIRED, SINGLE BLOCK $$$$$$$$$$$$$$$$$$$$$")

        if len(file_names) > 1: # Add files together if multiple blocks present
            # Add the fid data for the first and second file, then loop through and add the rest to the result
            print(f"$$$$$$$$$$$$$$$$$ ADDING {file_names[0]}/test{file_type} AND {file_names[1]}/test{file_type}  >>> added.fid $$$$$$$$$$$$$$$$$$$$$")
            os.system(f"addNMR -in1 {file_names[0]}/test{file_type} -in2 {file_names[1]}/test{file_type} -out added{file_type}")

            if len(file_names == 2): # Only need to do one addition, move added file to final directory
                # Put final added fid into the last file directory
                print(f"$$$$$$$$$$$$$$$$$ MOVING added{file_type} FROM CURRENT DIRECTORY TO FINAL ADDING DIRECTORY ./{file_names[-1]}/{file_names[-1]}_added{file_type} $$$$$$$$$$$$$$$$$$$$$")
                shutil.move(f"added{file_type}", f"./{file_names[-1]}/{file_names[-1]}_added{file_type}")

            if len(file_names) > 2: # Add remaining files to the initial two
                for fname in file_names[2:]:
                print(f"$$$$$$$$$$$$$$$$$ ADDING added{file_type} AND {fname}/test{file_type} >>> added.fid $$$$$$$$$$$$$$$$$$$$$")
                os.system(f"addNMR -in1 added{file_type} -in2 {fname}/test{file_type} -out added{file_type}")

                # Put final added fid into the last file directory
                print(f"$$$$$$$$$$$$$$$$$ MOVING added{file_type} FROM CURRENT DIRECTORY TO FINAL ADDING DIRECTORY ./{file_names[-1]}/{file_names[-1]}_added{file_type} $$$$$$$$$$$$$$$$$$$$$")
                shutil.move(f"added{file_type}", f"./{file_names[-1]}/{file_names[-1]}_added{file_type}")

# Process the titration point data
def processing(filename, nmrproc):
    os.system(f"{nmrproc} {fname}")