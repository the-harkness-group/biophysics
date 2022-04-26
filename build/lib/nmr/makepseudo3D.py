#!/usr/bin/env python3
import os 
import sys
import yaml

# This script allows you to concatenate multiple 2D files into single pseudo-3D file.
# The order of the Z-axis point is smaller to larger index number of **.ft2 file.
# By Yuki Toyama, edited by Rob Harkness to sort planes according to the concentration order of the titration points, low to high
# Requires input yaml file with list of experiment directory numbers and corresponding titration point concentrations

def main():
    # Read in parameters
    params = yaml.safe_load(open(sys.argv[1],'r'))
    # Stack separate titration point planes into single pseudo3D data cube
    makepseudo3D(params)

def makepseudo3D(params):

    # Get experiment directories and list of concentration points for titration
    files = params['Experiments']
    concentrations = params['Concentrations']
    sep = '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
    print(f"\n{sep}\nThe NMR titration planes that will be packaged into a pseudo3D cube are in directories: {files}\n{sep}")

    # Sort the experimental directory numbers and titration point concentrations in ascending order (low to high)
    # Use these to create a pseudo3D cube where the planes are ordered according to increasing ligand, for example
    zipped_files = zip(concentrations,files)
    sorted_zipped = sorted(zipped_files)
    tuples = zip(*sorted_zipped)
    sorted_concs, sorted_files = [list(tuple) for tuple in tuples]

    # Create new directory where the sorted experimental 2D planes from each directory will be copied to and renamed 
    # according to the default NMRPipe convention, i.e. test%03d.ft2 so that planes are test001.ft2 (concentration 1), 
    # test002.ft2 (concentration 2), etc.
    os.mkdir(f"./{params['Pseudo3D directory']}")
    nmrpipe_names = ['test{:03}.ft2'.format(x) for x in range(1,len(sorted_files)+1)]
    for f1,f2 in zip(sorted_files,nmrpipe_names): # Copy sorted files to new directory with new name according to sorted index
        os.system(f"cp ./{f1}/test.ft2 ./{params['Pseudo3D directory']}/{f2}")

    # Change headers of the nmrpipe-named 2D planes to fool NMRPipe into thinking they were collected in a pseudo3D format
    for f in nmrpipe_names:
        os.system(f"sethdr ./{params['Pseudo3D directory']}/{f} -ndim 3 -zLAB ID -zN {len(nmrpipe_names)} -zT {len(nmrpipe_names)}")
        os.system(f"showhdr ./{params['Pseudo3D directory']}/{f}")

    # Finally, created pseudo3D cube using the sorted 2D planes with modified headers
    os.system(f"xyz2pipe -in ./{params['Pseudo3D directory']}/test%03d.ft2 > ./{params['Pseudo3D directory']}/{params['Pseudo3D directory']}.ft3")

main()