#!/usr/bin/env python3
import os 
import sys
import yaml
import shutil
import nmr_water_ref

# Read parameters for converting temperature series to NMRPipe format
def main():
    # Get experiment numbers or names
    params = yaml.safe_load(open(sys.argv[1],'r'))
    # Run conversion script
    conversion(params)

# Convert NMR data to nmrpipe fid files
def conversion(params):
    # Convert each experiments data
    parent_dir = os.getcwd()
    conversion_script = params['Conversion script']

    ref_shift_dict = nmr_water_ref.water_tsp_shifts()

    for idx,expt in enumerate(params['Experiments']):
            try:
                O1_index = ref_shift_dict['Temperatures'].index(params['Temperatures'][idx])
                real_O1 = ref_shift_dict['Shifts'][O1_index]
                delta = real_O1 - params['Water'][idx]
                real_O1 = params['O1'][idx] + delta
                real_O2 = params['O2'][idx] + delta
                print(f"$$$$$$$$$$$$$$$$$ RUNNING BRUKER FID.COM {expt} $$$$$$$$$$$$$$$$$$$$$")
                os.system(f"{conversion_script} {expt} {real_O1} {real_O2}")
                print("\n")
            except:
                print("XXXXXXXXXXXXXXX THE EXPERIMENT TEMPERATURE IS NOT IN THE TSP LIST!!! XXXXXXXXXXXXXXXXX")
                print("\n")

main()
